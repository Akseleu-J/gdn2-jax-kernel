"""
gdn2_pipeline.py – trainable обёртка с fused Pallas backward (B1–B5).

Использует gdn2_fwd и gdn2_bwd, регистрирует custom_vjp.
Экспортирует gdn2_pallas_forward_trainable – основную функцию для обучения.
"""

from __future__ import annotations

from functools import partial
import jax
import jax.numpy as jnp

from gdn2_fwd import (
    BT,
    _reshape_to_chunks,
    _reshape_from_chunks,
    _sanitize,
    build_chunk_scores_pallas,
    wy_solve_pallas,
    recompute_wy_pallas,
    gdn2_inter_chunk_combine_with_state,
    gdn2_pallas_forward,
)
from gdn2_bwd import (
    gdn2_dhu_backward,
    dav_backward_pallas,
    wy_dqkg_backward_pallas,
    intra_backward_pallas,
    reverse_cumsum_bwd,
)

_HIGHEST = jax.lax.Precision.HIGHEST
_FINAL_CLIP = 1e4


def _final_sanitize(x):
    return jnp.nan_to_num(
        jnp.clip(x, -_FINAL_CLIP, _FINAL_CLIP),
        nan=0.0, posinf=_FINAL_CLIP, neginf=-_FINAL_CLIP
    )


def _reshape_in(t, bsz, n_chunks, H, D):
    return _reshape_to_chunks(t, bsz, n_chunks, H, D)


def _reshape_out(t):
    return _reshape_from_chunks(t)


def _build_dh_next_all(dh_all, dht):
    # dh_all: (B, H, n_chunks, D, D), dht: (B, H, D, D)
    shifted = dh_all[:, :, 1:]          # (B, H, n_chunks-1, D, D)
    dht_expanded = dht[:, :, None]      # (B, H, 1, D, D)
    return jnp.concatenate([shifted, dht_expanded], axis=2)


@partial(jax.custom_vjp, nondiff_argnums=(6,))
def _gdn2_core(q, k, v, w, b, g, scale, h0):
    return gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=h0)


def _gdn2_core_fwd(q, k, v, w, b, g, scale, h0):
    out, h_final = gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=h0)
    residuals = (q, k, v, w, b, g, h0)
    return (out, h_final), residuals


def _gdn2_core_bwd(scale, residuals, cotangents):
    q, k, v, w, b, g, h0 = residuals
    do, dh_final = cotangents

    bsz, L, H, D = q.shape
    n_chunks = L // BT

    # ---------- Повтор forward для получения промежуточных ----------
    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
    A = wy_solve_pallas(Akk)
    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
    _, _, h_pre_all, v_new_all = gdn2_inter_chunk_combine_with_state(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0
    )

    # ---------- Решейп входов в чанковую форму ----------
    g_r = _reshape_in(g, bsz, n_chunks, H, D)
    idx = jnp.arange(BT)
    tril_ones_bt = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    gc = jnp.einsum("ij,bhcjd->bhcid", tril_ones_bt, g_r, precision=_HIGHEST)

    q_r = _reshape_in(q, bsz, n_chunks, H, D)
    k_r = _reshape_in(k, bsz, n_chunks, H, D)
    b_r = _reshape_in(b, bsz, n_chunks, H, D)
    w_r = _reshape_in(w, bsz, n_chunks, H, D)
    v_r = _reshape_in(v, bsz, n_chunks, H, D)
    do_r = _reshape_in(do, bsz, n_chunks, H, D)

    # ---------- B2 ----------
    dAqk, dv_partial = dav_backward_pallas(Aqk, v_new_all, do_r)

    # ---------- B1 ----------
    dh_all, dh0, dv_all = gdn2_dhu_backward(
        do_r, dv_partial, w_pseudo, qg, kg, gc_last, scale, dht=dh_final
    )
    dh_next_all = _build_dh_next_all(dh_all, dh_final)

    # ---------- B3 ----------
    b3_out = wy_dqkg_backward_pallas(
        q_r, k_r, b_r, w_r, v_r, gc, A, Akk, h_pre_all, v_new_all,
        do_r, dv_all, dh_next_all, scale,
    )

    # ---------- B4 ----------
    dq4, dk4, db4, dgc4 = intra_backward_pallas(dAqk, b3_out["dAkk"], q, k, b, g, scale)

    # ---------- B5 ----------
    dgc_total = b3_out["dgc"] + dgc4
    dg_raw = reverse_cumsum_bwd(dgc_total, chunk_size=BT)

    # ---------- Сборка градиентов и решейп обратно ----------
    dq = _reshape_out(b3_out["dq"] + dq4)
    dk = _reshape_out(b3_out["dk"] + dk4)
    db = _reshape_out(b3_out["db"] + db4)
    dw = _reshape_out(b3_out["dw"])
    dv = _reshape_out(b3_out["dv_raw"])
    dg = _reshape_out(dg_raw)

    # ---------- Финальное отсечение и приведение dtype ----------
    dq = _final_sanitize(dq).astype(q.dtype)
    dk = _final_sanitize(dk).astype(k.dtype)
    db = _final_sanitize(db).astype(b.dtype)
    dw = _final_sanitize(dw).astype(w.dtype)
    dv = _final_sanitize(dv).astype(v.dtype)
    dg = _final_sanitize(dg).astype(g.dtype)
    dh0 = _final_sanitize(dh0).astype(h0.dtype)

    return dq, dk, dv, dw, db, dg, dh0


_gdn2_core.defvjp(_gdn2_core_fwd, _gdn2_core_bwd)


def gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale, h0=None):
    """
    Trainable Gated DeltaNet2 forward с fused Pallas backward.

    Args:
        q, k, v, w, b, g: тензоры формы (batch, seq_len, heads, d_head)
        scale: коэффициент масштабирования (скаляр)
        h0: начальное состояние (batch, heads, d_head, d_head), если None – нулевое

    Returns:
        out: выходной тензор (batch, seq_len, heads, d_head)
        h_final: финальное состояние (batch, heads, d_head, d_head)
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    return _gdn2_core(q, k, v, w, b, g, scale, h0)
