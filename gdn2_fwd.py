"""
gdn2_fwd.py – все forward-ядра для Gated DeltaNet2 (Pallas/TPU).

Содержит:
- Константы: BT, BC, N_SUB, MB, CLIP.
- Утилиты: _sanitize, _stage_diag, _reshape_to_chunks, _reshape_from_chunks.
- Kernel A: build_chunk_scores_pallas
- Kernel B: wy_solve_pallas
- Kernel C: recompute_wy_pallas
- Kernel D: gdn2_inter_chunk_combine, _with_state,
           gdn2_pallas_forward, gdn2_pallas_forward_with_residuals.

Использование:
    from gdn2_fwd import gdn2_pallas_forward
    out, h_final = gdn2_pallas_forward(q, k, v, w, b, g, scale)

Для диагностики включите переменную окружения GDN2_FWD_DIAG=1.
"""

from __future__ import annotations

import os
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# ---------- Константы ----------
BT = 256                      # размер чанка
BC = 128                      # размер субблока (BT / N_SUB)
N_SUB = BT // BC              # = 2 (используется в Kernel A и B4)
MB = 16                       # микро-блок для блочного решения (Kernel B)
CLIP = 1e4                    # порог для отсечения
_HIGHEST = jax.lax.Precision.HIGHEST

# Диагностика (включается через env)
_GDN2_FWD_DIAG = os.environ.get("GDN2_FWD_DIAG", "0") == "1"
_LARGE_THRESHOLD = 1e6


# ---------- Утилиты ----------
def _sanitize(x):
    """Отсечь и заменить nan/inf на 0."""
    return jnp.nan_to_num(jnp.clip(x, -CLIP, CLIP), nan=0.0, posinf=CLIP, neginf=-CLIP)


def _stage_diag(tag: str, x):
    """Диагностический вывод, если GDN2_FWD_DIAG=1."""
    if not _GDN2_FWD_DIAG:
        return x
    finite_mask = jnp.isfinite(x)
    all_finite = jnp.all(finite_mask)
    n_nonfinite = jnp.sum(jnp.logical_not(finite_mask))
    safe_x = jnp.where(finite_mask, x, 0.0)
    max_abs = jnp.max(jnp.abs(safe_x))

    def _report_nonfinite():
        jax.debug.print(
            "[GDN2-FWD-DIAG] ⚠️ non-finite на выходе " + tag +
            ": n_nonfinite={n}  max_abs(конечная часть)={m:.3e}",
            n=n_nonfinite, m=max_abs,
        )

    def _report_large():
        jax.debug.print(
            "[GDN2-FWD-DIAG] 🔶 подозрительно большая величина на выходе " + tag +
            " (всё ещё конечная, но уже похоже на предвестник): max_abs={m:.3e}",
            m=max_abs,
        )

    jax.lax.cond(
        jnp.logical_not(all_finite),
        _report_nonfinite,
        lambda: jax.lax.cond(max_abs > _LARGE_THRESHOLD, _report_large, lambda: None),
    )
    return x


def _reshape_to_chunks(t, bsz, n_chunks, H, D):
    """Преобразовать (B,L,H,D) -> (B,H,n_chunks,BT,D)."""
    t = t.reshape(bsz, n_chunks, BT, H, D)
    return jnp.moveaxis(t, (1, 3), (2, 1))


def _reshape_from_chunks(t):
    """Обратно: (B,H,n_chunks,BT,D) -> (B,L,H,D)."""
    bsz, H, n_chunks, _BT, D = t.shape
    t2 = jnp.moveaxis(t, (1, 2, 3), (3, 1, 2))
    return t2.reshape(bsz, n_chunks * BT, H, D)


# ---------- Kernel A: построение Aqk, Akk ----------
def _weighted_pair_sum(a_i, edecay, b_j):
    tmp = a_i[:, None, :] * edecay
    tmp = tmp * b_j[None, :, :]
    return jnp.sum(tmp, axis=-1)


def _kernel_a_body(q_ref, k_ref, b_ref, g_ref, aqk_ref, akk_ref, *, scale):
    q_full = q_ref[0, 0, 0].astype(jnp.float32)
    k_full = k_ref[0, 0, 0].astype(jnp.float32)
    b_full = b_ref[0, 0, 0].astype(jnp.float32)
    g_raw = g_ref[0, 0, 0].astype(jnp.float32)

    bt_idx = jnp.arange(BT)
    tril_ones_bt = (bt_idx[:, None] >= bt_idx[None, :]).astype(jnp.float32)
    gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)

    aqk_ref[0, 0, 0] = jnp.zeros((BT, BT), dtype=jnp.float32)
    akk_ref[0, 0, 0] = jnp.zeros((BT, BT), dtype=jnp.float32)

    for si in range(N_SUB):
        for sj in range(si + 1):
            i0, i1 = si * BC, (si + 1) * BC
            j0, j1 = sj * BC, (sj + 1) * BC

            q_i = q_full[i0:i1]
            k_i = k_full[i0:i1]
            k_j = k_full[j0:j1]
            b_i = b_full[i0:i1]
            gc_i = gc[i0:i1]
            gc_j = gc[j0:j1]

            decay_diff = gc_i[:, None, :] - gc_j[None, :, :]
            edecay = jnp.exp(jnp.clip(decay_diff, -20.0, 20.0))

            aqk_blk = scale * _weighted_pair_sum(q_i, edecay, k_j)
            bk_i = b_i * k_i
            akk_blk = _weighted_pair_sum(bk_i, edecay, k_j)

            if si == sj:
                idx = jnp.arange(BC)
                causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
                strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
                aqk_blk = aqk_blk * causal
                akk_blk = akk_blk * strict

            aqk_ref[0, 0, 0, i0:i1, j0:j1] = _sanitize(aqk_blk)
            akk_ref[0, 0, 0, i0:i1, j0:j1] = _sanitize(akk_blk)


def build_chunk_scores_pallas(q, k, b, g, scale):
    """Возвращает Aqk, Akk размером (B,H,n_chunks,BT,BT)."""
    bsz, L, H, D = q.shape
    assert D == 128, f"Kernel A assumes d_head=128; got D={D}."
    assert L % BT == 0, f"L={L} must be divisible by BT={BT}."
    n_chunks = L // BT

    q_r, k_r, b_r, g_r = map(
        lambda t: _reshape_to_chunks(t, bsz, n_chunks, H, D),
        (q, k, b, g)
    )

    grid = (bsz, H, n_chunks)
    in_spec = pl.BlockSpec((1, 1, 1, BT, D), lambda i, h, c: (i, h, c, 0, 0))
    out_spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))

    aqk, akk = pl.pallas_call(
        lambda *refs: _kernel_a_body(*refs, scale=scale),
        grid=grid,
        in_specs=[in_spec, in_spec, in_spec, in_spec],
        out_specs=[out_spec, out_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, BT), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, BT), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 * 1024),
    )(q_r, k_r, b_r, g_r)

    return aqk, akk


# ---------- Kernel B: WY-решение (I + Akk)^{-1} ----------
def _micro_forward_substitution(T_mb):
    """Forward substitution для одного микро-блока MB×MB."""
    idx = jnp.arange(MB)

    def body(i, A):
        onehot_i = (idx == i).astype(jnp.float32)
        t_row = jnp.sum(T_mb * onehot_i[:, None], axis=0)
        contrib = jnp.sum(t_row[:, None] * A, axis=0)
        new_row = onehot_i - contrib
        new_row = _sanitize(new_row)
        mask_col = onehot_i[:, None]
        A = A * (1.0 - mask_col) + mask_col * new_row[None, :]
        return A

    A0 = jnp.zeros((MB, MB), dtype=jnp.float32)
    return jax.lax.fori_loop(0, MB, body, A0)


def _block_solve(T_full):
    """Блочная forward-подстановка для (BC,BC) строго нижнетреугольной."""
    N_MICRO = BC // MB
    blocks = [[None] * N_MICRO for _ in range(N_MICRO)]

    for m in range(N_MICRO):
        T_mm = T_full[m * MB:(m + 1) * MB, m * MB:(m + 1) * MB]
        A_mm = _sanitize(_micro_forward_substitution(T_mm))
        blocks[m][m] = A_mm

        for n in range(m - 1, -1, -1):
            acc = jnp.zeros((MB, MB), dtype=jnp.float32)
            for k in range(n, m):
                T_mk = T_full[m * MB:(m + 1) * MB, k * MB:(k + 1) * MB]
                A_kn = blocks[k][n]
                contrib = jnp.dot(T_mk, A_kn, precision=_HIGHEST)
                acc = _sanitize(acc + contrib)
            A_mn = -jnp.dot(A_mm, acc, precision=_HIGHEST)
            A_mn = _sanitize(A_mn)
            blocks[m][n] = A_mn

    rows = []
    for m in range(N_MICRO):
        row_blocks = []
        for n in range(N_MICRO):
            if n > m:
                row_blocks.append(jnp.zeros((MB, MB), dtype=jnp.float32))
            else:
                row_blocks.append(blocks[m][n])
        rows.append(jnp.concatenate(row_blocks, axis=1))
    return jnp.concatenate(rows, axis=0)


def _kernel_b_body(akk_ref, a_ref):
    Akk = akk_ref[0, 0, 0].astype(jnp.float32)
    # Делим на два субблока (N_SUB=2)
    T00 = Akk[0:BC, 0:BC]
    T11 = Akk[BC:2*BC, BC:2*BC]
    T10 = Akk[BC:2*BC, 0:BC]

    A00 = _block_solve(T00)
    A11 = _block_solve(T11)

    tmp = jnp.dot(T10, A00, precision=_HIGHEST)
    tmp = _sanitize(tmp)
    A10 = -jnp.dot(A11, tmp, precision=_HIGHEST)
    A10 = _sanitize(A10)

    a_ref[0, 0, 0] = jnp.zeros((BT, BT), dtype=jnp.float32)
    a_ref[0, 0, 0, 0:BC, 0:BC] = A00
    a_ref[0, 0, 0, BC:2*BC, 0:BC] = A10
    a_ref[0, 0, 0, BC:2*BC, BC:2*BC] = A11


def wy_solve_pallas(Akk):
    """Решает A = (I + Akk)^{-1} для каждого чанка."""
    bsz, H, n_chunks = Akk.shape[:3]
    assert Akk.shape[-2:] == (BT, BT)
    grid = (bsz, H, n_chunks)
    spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))
    A = pl.pallas_call(
        _kernel_b_body,
        grid=grid,
        in_specs=[spec],
        out_specs=spec,
        out_shape=jax.ShapeDtypeStruct(Akk.shape, jnp.float32),
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=96 * 1024 * 1024),
    )(Akk)
    return A


# ---------- Kernel C: пересчёт w_pseudo, u, kg, qg, gc_last ----------
def _kernel_c_body(q_ref, k_ref, v_ref, w_ref, b_ref, g_ref, a_ref,
                   w_pseudo_ref, u_ref, kg_ref, qg_ref, gc_last_ref):
    q = q_ref[0, 0, 0].astype(jnp.float32)
    k = k_ref[0, 0, 0].astype(jnp.float32)
    v = v_ref[0, 0, 0].astype(jnp.float32)
    w = w_ref[0, 0, 0].astype(jnp.float32)
    b = b_ref[0, 0, 0].astype(jnp.float32)
    g_raw = g_ref[0, 0, 0].astype(jnp.float32)
    A = a_ref[0, 0, 0].astype(jnp.float32)

    bt_idx = jnp.arange(BT)
    tril_ones_bt = (bt_idx[:, None] >= bt_idx[None, :]).astype(jnp.float32)
    gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)

    kb_decayed = b * k * jnp.exp(gc)
    w_pseudo = jnp.dot(A, kb_decayed, precision=_HIGHEST)
    u = jnp.dot(A, w * v, precision=_HIGHEST)
    w_pseudo = _sanitize(w_pseudo)
    u = _sanitize(u)

    gc_last_row = gc[BT - 1]
    kg = k * jnp.exp(gc_last_row[None, :] - gc)
    qg = q * jnp.exp(gc)
    kg = _sanitize(kg)
    qg = _sanitize(qg)
    gc_last_row = _sanitize(gc_last_row)

    w_pseudo_ref[0, 0, 0] = w_pseudo
    u_ref[0, 0, 0] = u
    kg_ref[0, 0, 0] = kg
    qg_ref[0, 0, 0] = qg
    gc_last_ref[0, 0, 0, 0] = gc_last_row


def recompute_wy_pallas(q, k, v, w, b, g, A):
    """Вычисляет w_pseudo, u, kg, qg, gc_last для каждого чанка."""
    bsz, L, H, D = q.shape
    assert L % BT == 0
    n_chunks = L // BT

    def reshape_in(t):
        return _reshape_to_chunks(t, bsz, n_chunks, H, D)

    q_r, k_r, v_r, w_r, b_r, g_r = map(reshape_in, (q, k, v, w, b, g))

    grid = (bsz, H, n_chunks)
    io_spec = pl.BlockSpec((1, 1, 1, BT, D), lambda i, h, c: (i, h, c, 0, 0))
    a_spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))
    gclast_spec = pl.BlockSpec((1, 1, 1, 1, D), lambda i, h, c: (i, h, c, 0, 0))

    w_pseudo, u, kg, qg, gc_last = pl.pallas_call(
        _kernel_c_body,
        grid=grid,
        in_specs=[io_spec, io_spec, io_spec, io_spec, io_spec, io_spec, a_spec],
        out_specs=[io_spec, io_spec, io_spec, io_spec, gclast_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, 1, D), jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
    )(q_r, k_r, v_r, w_r, b_r, g_r, A)

    gc_last = gc_last.reshape(bsz, H, n_chunks, D)
    return w_pseudo, u, kg, qg, gc_last


# ---------- Kernel D: inter‑chunk комбинация и полный forward ----------
def _sanitize_h0(h0):
    return jnp.nan_to_num(jnp.clip(h0, -CLIP, CLIP), nan=0.0, posinf=CLIP, neginf=-CLIP)


def gdn2_inter_chunk_combine(Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=None, debug_tag=""):
    """Без сохранения промежуточных состояний (для простого forward)."""
    bsz, H, n_chunks, _BT, D = w_pseudo.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    h0 = _sanitize_h0(h0)

    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (Aqk, w_pseudo, u, kg, qg, gc_last))

    def step(h_pre, inputs):
        Aqk_c, w_pseudo_c, u_c, kg_c, qg_c, gclast_c = inputs
        wh = jnp.einsum("bhid,bhdv->bhiv", w_pseudo_c, h_pre, precision=_HIGHEST)
        v_new = u_c - wh
        qh = jnp.einsum("bhid,bhdv->bhiv", qg_c, h_pre, precision=_HIGHEST)
        intra = jnp.einsum("bhij,bhjv->bhiv", Aqk_c, v_new, precision=_HIGHEST)
        o_c = scale * qh + intra

        decay_h = jnp.exp(gclast_c)[..., None]
        write = jnp.einsum("bhid,bhiv->bhdv", kg_c, v_new, precision=_HIGHEST)
        h_new = h_pre * decay_h + write
        h_new = _sanitize(h_new)
        o_c = _sanitize(o_c)
        return h_new, o_c

    h_final, o_scanned = jax.lax.scan(step, h0, to_scan)
    h_final = _stage_diag(f"{debug_tag}:kernel_D_h_final", h_final)
    o = jnp.moveaxis(o_scanned, 0, 2)
    o = _stage_diag(f"{debug_tag}:kernel_D_o", o)
    return o, h_final


def gdn2_inter_chunk_combine_with_state(Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=None, debug_tag=""):
    """То же, но дополнительно возвращает h_pre_all и v_new_all для backward."""
    bsz, H, n_chunks, _BT, D = w_pseudo.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    h0 = _sanitize_h0(h0)

    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (Aqk, w_pseudo, u, kg, qg, gc_last))

    def step(h_pre, inputs):
        Aqk_c, w_pseudo_c, u_c, kg_c, qg_c, gclast_c = inputs
        wh = jnp.einsum("bhid,bhdv->bhiv", w_pseudo_c, h_pre, precision=_HIGHEST)
        v_new = u_c - wh
        qh = jnp.einsum("bhid,bhdv->bhiv", qg_c, h_pre, precision=_HIGHEST)
        intra = jnp.einsum("bhij,bhjv->bhiv", Aqk_c, v_new, precision=_HIGHEST)
        o_c = scale * qh + intra

        decay_h = jnp.exp(gclast_c)[..., None]
        write = jnp.einsum("bhid,bhiv->bhdv", kg_c, v_new, precision=_HIGHEST)
        h_new = h_pre * decay_h + write
        h_new = _sanitize(h_new)
        o_c = _sanitize(o_c)
        return h_new, (o_c, h_pre, v_new)

    h_final, (o_scanned, h_pre_all, v_new_all) = jax.lax.scan(step, h0, to_scan)
    h_final = _stage_diag(f"{debug_tag}:kernel_D_h_final", h_final)
    o = jnp.moveaxis(o_scanned, 0, 2)
    o = _stage_diag(f"{debug_tag}:kernel_D_o", o)
    return o, h_final, h_pre_all, v_new_all


def gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=None, debug_tag=""):
    """Полный forward: A -> B -> C -> D. Возвращает o (B,L,H,D) и h_final (B,H,D,D)."""
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
    Aqk = _stage_diag(f"{debug_tag}:kernel_A_Aqk", Aqk)
    Akk = _stage_diag(f"{debug_tag}:kernel_A_Akk", Akk)

    A = wy_solve_pallas(Akk)
    A = _stage_diag(f"{debug_tag}:kernel_B_wy_inverse_A", A)

    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
    w_pseudo = _stage_diag(f"{debug_tag}:kernel_C_w_pseudo", w_pseudo)
    u = _stage_diag(f"{debug_tag}:kernel_C_u", u)
    kg = _stage_diag(f"{debug_tag}:kernel_C_kg", kg)
    qg = _stage_diag(f"{debug_tag}:kernel_C_qg", qg)

    o_chunks, h_final = gdn2_inter_chunk_combine(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0, debug_tag=debug_tag
    )
    n_chunks = L // BT
    o = _reshape_from_chunks(o_chunks)
    return o, h_final


def gdn2_pallas_forward_with_residuals(q, k, v, w, b, g, scale, h0=None, debug_tag=""):
    """Как forward, но возвращает residuals для backward (Aqk, h_pre_all, v_new_all и др.)."""
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
    Aqk = _stage_diag(f"{debug_tag}:kernel_A_Aqk", Aqk)
    Akk = _stage_diag(f"{debug_tag}:kernel_A_Akk", Akk)

    A = wy_solve_pallas(Akk)
    A = _stage_diag(f"{debug_tag}:kernel_B_wy_inverse_A", A)

    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
    w_pseudo = _stage_diag(f"{debug_tag}:kernel_C_w_pseudo", w_pseudo)
    u = _stage_diag(f"{debug_tag}:kernel_C_u", u)
    kg = _stage_diag(f"{debug_tag}:kernel_C_kg", kg)
    qg = _stage_diag(f"{debug_tag}:kernel_C_qg", qg)

    o_chunks, h_final, h_pre_all, v_new_all = gdn2_inter_chunk_combine_with_state(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0, debug_tag=debug_tag
    )
    n_chunks = L // BT
    o = _reshape_from_chunks(o_chunks)

    residuals = {
        "Aqk": Aqk, "Akk": Akk, "A": A,
        "h_pre_all": h_pre_all, "v_new_all": v_new_all,
        "w_pseudo": w_pseudo, "u": u, "kg": kg, "qg": qg, "gc_last": gc_last,
    }
    return o, h_final, residuals
