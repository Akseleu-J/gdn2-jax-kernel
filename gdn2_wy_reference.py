"""
Milestone 1 -- pure JAX WY-chunked reference for GDN-2, plus a token-serial
ground-truth reference, matching the conventions used in model.py.

Every formula here is transliterated from the uploaded NVIDIA/fla source
(no invented algebra):
  - token-serial recurrence  <- fused_recurrent_gdn2_fwd_kernel (fused_recurrent_gdn2.py, L193-244)
  - Aqk / Akk intra-chunk    <- chunk_gdn2_fwd_kernel_intra_token_parallel / _intra_sub_chunk (chunk_gdn2.py)
  - A = (I+Akk)^-1 WY solve  <- chunk_gdn2_fwd_kernel_inter_solve_fused (chunk_gdn2.py)
  - w_pseudo / u / kg / qg   <- recompute_w_u_fwd_gdn2_kernel (chunk_gdn2.py, L698-790)
  - inter-chunk state update <- chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (chunk_kda.py, L977-1263)
  - output combine           <- derived by matching per-token unrolling of the
                                 inter-chunk recurrence against the intra-chunk
                                 Aqk/qg/kg terms (verified numerically to match
                                 the token-serial reference to float64 machine
                                 precision across 6 shape/edge-case configs --
                                 see verify_wy_numpy.py).

Shape convention (matches model.py's GatedDeltaNet2J): (b, l, n_heads, d_head)
for q, k, v, b_gate (erase), w_gate (write), g (raw per-token log-decay).
d_head is used for both the key axis and the value axis (single head dim),
same as the rest of model.py.

Precision: WY solve done in float32 (Milestone-0 decision: fp32 first, bf16
optimization later). Inputs/outputs can be any dtype; internal accumulation
is float32.

IMPORTANT (TPU-specific, found empirically on v5e): XLA on TPU runs float32
matmuls at REDUCED internal precision by default (bfloat16-based passes),
even when both operands are literally float32 arrays. This silently degraded
WY-solve accuracy to ~1e-3 relative error on real TPU hardware, vs ~1e-15 in
float64 NumPy for the IDENTICAL algebra (confirmed by direct A/B: the earlier
jnp.linalg.inv-based version reproduced this gap; forcing HIGHEST precision
and switching to an explicit forward-substitution solve below is the fix).
All dot/einsum calls pass precision=HIGHEST explicitly -- do not remove this
when porting into a jax.jit'd training step. As a belt-and-braces backup you
can also set jax.config.update("jax_default_matmul_precision", "highest")
globally in train.py, but do not rely on that alone: some jnp.linalg.* paths
(e.g. the inv() this file no longer uses) have historically not obeyed that
global on TPU, which is exactly the bug we hit.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST


def _wy_inverse(Akk):
    """Solve A = (I + Akk)^{-1} for strictly-lower-triangular Akk via row-by-row
    forward substitution -- mirrors the explicit elimination loop in
    chunk_gdn2_fwd_kernel_intra_sub_chunk / _inter_solve_fused (Kernels 2/3),
    rather than a generic matrix inverse (jnp.linalg.inv proved LESS accurate
    on TPU for this precision-sensitive step -- see module docstring).

    Akk: (..., C, C) strictly lower triangular (zero diagonal and above).
    Returns A: (..., C, C).
    """
    C = Akk.shape[-1]
    eye = jnp.eye(C, dtype=Akk.dtype)
    batch_shape = Akk.shape[:-2]

    def row_step(A_rows, i):
        # A_rows: (..., C, C); rows < i are already correct, rows >= i are zero.
        t_row = jnp.take(Akk, i, axis=-2)  # (..., C) == Akk[..., i, :]
        contrib = jnp.einsum("...j,...jk->...k", t_row, A_rows, precision=_HIGHEST)
        new_row = eye[i] - contrib  # (..., C); eye[i] broadcasts over leading batch dims
        A_rows = jax.lax.dynamic_update_slice_in_dim(A_rows, new_row[..., None, :], i, axis=-2)
        return A_rows, None

    A0 = jnp.zeros(batch_shape + (C, C), dtype=Akk.dtype)
    A_final, _ = jax.lax.scan(row_step, A0, jnp.arange(C))
    return A_final


def gdn2_token_serial_reference(q, k, v, g, b, w, scale, h0=None):
    """Ground truth: fused_recurrent_gdn2_fwd_kernel, unrolled token-by-token
    via jax.lax.scan. Use this ONLY to validate the chunked kernel -- it is
    O(T) sequential matrix ops and not meant for production training/inference.

    q, k, b: (B, L, H, D)   v, w: (B, L, H, Dv)   g: (B, L, H, D) raw log-decay.
    h0: optional (B, H, D, Dv) initial state.
    Returns o: (B, L, H, Dv), h_final: (B, H, D, Dv).
    """
    bsz, L, H, D = q.shape
    Dv = v.shape[-1]
    dtype = q.dtype

    alpha = jnp.exp(g.astype(jnp.float32))  # (B, L, H, D)

    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, Dv), dtype=jnp.float32)

    def step(h, inputs):
        q_t, k_t, v_t, alpha_t, b_t, w_t = inputs  # each (B, H, *)
        h = h * alpha_t[..., :, None]                                  # (B,H,D,Dv)
        bk_t = (b_t * k_t).astype(jnp.float32)                          # (B,H,D)
        erase = jnp.einsum("bhd,bhdv->bhv", bk_t, h, precision=_HIGHEST)
        v_new = (w_t * v_t).astype(jnp.float32) - erase                   # (B,H,Dv)
        h = h + jnp.einsum("bhd,bhv->bhdv", k_t.astype(jnp.float32), v_new, precision=_HIGHEST)
        o_t = jnp.einsum("bhdv,bhd->bhv", h, (q_t * scale).astype(jnp.float32), precision=_HIGHEST)
        return h, o_t

    to_scan = tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, alpha, b, w))
    h_final, o_scanned = jax.lax.scan(step, h0, to_scan)
    o = jnp.moveaxis(o_scanned, 0, 1).astype(dtype)  # (B, L, H, Dv)
    return o, h_final


def _build_chunk_wy(q_c, k_c, v_c, g_raw_c, b_c, w_c, scale):
    """One chunk's intra-chunk WY factors. All inputs (B, C, H, D[/Dv]).

    Returns Aqk (B,H,C,C), w_pseudo (B,C,H,D), u (B,C,H,Dv),
            kg (B,C,H,D), qg (B,C,H,D), gc_last (B,H,D).
    """
    C = q_c.shape[1]
    f32 = jnp.float32

    gc = jnp.cumsum(g_raw_c.astype(f32), axis=1)  # chunk-local cumsum, inclusive (chunk_local_cumsum)

    gc_bhcd = jnp.moveaxis(gc, 2, 1)  # (B,H,C,D)
    decay_diff = gc_bhcd[:, :, :, None, :] - gc_bhcd[:, :, None, :, :]  # (B,H,C,C,D)
    # ФИКС (пользователь, до инцидента 710+): decay_diff теоретически всегда
    # <=0 для причинных пар (t>=j, decay<=1), но численно (bf16-округления
    # выше по графу, экстремальные decay_a) может просачиваться и в плюс --
    # exp без клипа на это даёт inf. Без этого клипа обучение падало сразу
    # же (не через 700 шагов, а немедленно) -- это первый, самый важный
    # рубеж защиты, nan_to_num на Aqk/Akk/A ниже -- уже вторая линия.
    edecay = jnp.exp(jnp.clip(decay_diff, -20.0, 20.0))

    causal = jnp.tril(jnp.ones((C, C), dtype=f32))          # j<=i
    strict = jnp.tril(jnp.ones((C, C), dtype=f32), k=-1)    # j<i

    q_bhcd = jnp.moveaxis(q_c, 2, 1).astype(f32)  # (B,H,C,D)
    k_bhcd = jnp.moveaxis(k_c, 2, 1).astype(f32)
    b_bhcd = jnp.moveaxis(b_c, 2, 1).astype(f32)

    Aqk = scale * jnp.einsum("bhid,bhijd,bhjd->bhij", q_bhcd, edecay, k_bhcd, precision=_HIGHEST) * causal
    bk_bhcd = b_bhcd * k_bhcd
    Akk = jnp.einsum("bhid,bhijd,bhjd->bhij", bk_bhcd, edecay, k_bhcd, precision=_HIGHEST) * strict  # (B,H,C,C)

    # ФИКС: этот путь (не Pallas Kernel A/B/C!) -- единственный, который
    # реально исполняется в backward (через jax.vjp в kernel_trainable.py),
    # т.к. custom_vjp считает градиент через ЭТОТ чистый JAX референс, а не
    # через forward Pallas-кернелы. Санитизация в kernel_a_scores.py (forward)
    # НЕ защищает backward -- это два независимых пути вычисления одной
    # математики. Инцидент на реальном обучении (шаг 710+, non-finite delta
    # в gdn2 block4/layer14, затем non-finite и в backward) показал, что без
    # клипа здесь Akk/A могут уйти в нестабильный режим по мере дрейфа весов,
    # так же как это уже случалось со старым associative_scan-путём
    # (см. Frobenius-clip в его _combine).
    Aqk = jnp.nan_to_num(Aqk, nan=0.0, posinf=1e4, neginf=-1e4)
    Akk = jnp.nan_to_num(Akk, nan=0.0, posinf=1e4, neginf=-1e4)

    A = _wy_inverse(Akk)  # (B,H,C,C) -- explicit forward substitution, not jnp.linalg.inv
    A = jnp.nan_to_num(A, nan=0.0, posinf=1e4, neginf=-1e4)

    kb_decayed = (b_c.astype(f32) * k_c.astype(f32)) * jnp.exp(gc)  # (B,C,H,D)
    w_pseudo = jnp.einsum("bhij,bjhd->bihd", A, kb_decayed, precision=_HIGHEST)          # (B,C,H,D)
    u = jnp.einsum("bhij,bjhv->bihv", A, (w_c * v_c).astype(f32), precision=_HIGHEST)    # (B,C,H,Dv)
    w_pseudo = jnp.nan_to_num(w_pseudo, nan=0.0, posinf=1e4, neginf=-1e4)
    u = jnp.nan_to_num(u, nan=0.0, posinf=1e4, neginf=-1e4)

    gc_last = gc[:, -1]  # (B,H,D)
    kg = k_c.astype(f32) * jnp.exp(gc_last[:, None] - gc)  # (B,C,H,D)
    qg = q_c.astype(f32) * jnp.exp(gc)                       # (B,C,H,D)

    return Aqk, w_pseudo, u, kg, qg, gc_last


def gdn2_chunked_wy_reference(q, k, v, g, b, w, scale, chunk_size, h0=None):
    """Full chunked WY forward. Pure JAX (no Pallas yet) -- Milestone 1.

    q, k, b: (B, L, H, D)   v, w: (B, L, H, Dv)   g: (B, L, H, D) raw log-decay.
    L must be divisible by chunk_size (pad upstream if not -- same requirement
    as GatedDeltaNet2J.__call__'s existing chunk_size check in model.py).
    h0: optional (B, H, D, Dv) initial state.
    Returns o: (B, L, H, Dv), h_final: (B, H, D, Dv).
    """
    bsz, L, H, D = q.shape
    Dv = v.shape[-1]
    dtype = q.dtype
    assert L % chunk_size == 0, (
        f"seq_len={L} must be divisible by chunk_size={chunk_size} "
        f"(same constraint as the existing associative_scan GDN-2 path)."
    )
    n_chunks = L // chunk_size

    def to_chunks(t):
        shp = t.shape
        t = t.reshape(bsz, n_chunks, chunk_size, *shp[2:])
        return jnp.moveaxis(t, 1, 0)  # (n_chunks, B, C, ...)

    q_ch, k_ch, v_ch, g_ch, b_ch, w_ch = map(to_chunks, (q, k, v, g, b, w))

    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, Dv), dtype=jnp.float32)

    def chunk_step(h_pre, inputs):
        q_c, k_c, v_c, g_c, b_c, w_c = inputs
        Aqk, w_pseudo, u, kg, qg, gc_last = _build_chunk_wy(q_c, k_c, v_c, g_c, b_c, w_c, scale)

        # v_new = u - w_pseudo @ h_pre  (WY correction using the PRE-chunk state)
        wh = jnp.einsum("bihd,bhdv->bihv", w_pseudo, h_pre, precision=_HIGHEST)  # (B,C,H,Dv)
        v_new = u - wh

        # output: inter-chunk (q decayed @ h_pre) + intra-chunk (Aqk @ v_new)
        qh = jnp.einsum("bihd,bhdv->bihv", qg, h_pre, precision=_HIGHEST)  # (B,C,H,Dv)
        v_new_bhcv = jnp.moveaxis(v_new, 2, 1)  # (B,H,C,Dv)
        intra = jnp.einsum("bhij,bhjv->bhiv", Aqk, v_new_bhcv, precision=_HIGHEST)
        intra = jnp.moveaxis(intra, 1, 2)  # (B,C,H,Dv)
        # ФИКС: НЕ кастуем к dtype=q.dtype здесь. gdn2_pallas_forward (Pallas
        # Kernel D) никогда не кастует свой выход -- всегда float32. Если q
        # приходит bfloat16 (реальная модель), а этот референс (используется
        # ТОЛЬКО внутри _gdn2_core_bwd для jax.vjp) кастует в bf16, то
        # forward-путь и backward-референс-путь получают РАЗНЫЙ dtype для
        # логически одного и того же тензора -- custom_vjp с этим падает
        # ("unexpected JAX type... expected bfloat16... got float32", found
        # on real training run). Даункаст в bf16 делает вызывающий код
        # (GatedDeltaNet2J, после RMSNorm), не этот референс.
        o_c = scale * qh + intra

        # inter-chunk state update: h_pre <- h_pre * exp(gc_last) + kg^T @ v_new
        decay_h = jnp.exp(gc_last)[..., None]  # (B,H,D,1)
        write = jnp.einsum("bihd,bihv->bhdv", kg, v_new, precision=_HIGHEST)  # (B,H,D,Dv)
        h_new = h_pre * decay_h + write
        # ФИКС: тот же рубеж защиты, что уже стоит в Pallas Kernel D
        # (kernel_d_pipeline.py) -- этот путь исполняется в backward, у него
        # своя копия состояния, отдельная от Pallas-forward, поэтому нужен
        # свой собственный клип, а не общий с forward.
        h_new = jnp.nan_to_num(jnp.clip(h_new, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
        o_c = jnp.nan_to_num(o_c, nan=0.0, posinf=1e4, neginf=-1e4)

        return h_new, o_c

    # NOTE: without this, XLA's reverse-mode autodiff (used by
    # kernel_trainable.py's custom_vjp backward) keeps EVERY chunk's
    # (B,H,C,C,D)-sized intermediates (from _build_chunk_wy's decay_diff/
    # edecay, ~1.6GB per chunk at B=8,H=6,C=256,D=128) alive simultaneously
    # for all n_chunks steps of the scan -- confirmed on real v5e-8: 105GB
    # HLO temporaries requested for B=8,L=4096 (16 chunks) vs 15.75GB
    # available, i.e. ~16x too much, matching "all chunks held at once"
    # instead of "one chunk recomputed at a time". jax.checkpoint makes
    # autodiff recompute this chunk's forward during backward instead of
    # storing it, matching the same pattern already used in model.py's
    # associative_scan _chunk_step.
    chunk_step = jax.checkpoint(chunk_step)

    h_final, o_scanned = jax.lax.scan(
        chunk_step, h0, (q_ch, k_ch, v_ch, g_ch, b_ch, w_ch)
    )
    # o_scanned: (n_chunks, B, C, H, Dv) -> (B, L, H, Dv)
    o = jnp.moveaxis(o_scanned, 0, 1).reshape(bsz, L, H, Dv)
    return o, h_final
