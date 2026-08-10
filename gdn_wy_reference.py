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
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


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
        erase = jnp.einsum("bhd,bhdv->bhv", bk_t, h)                     # (B,H,Dv)
        v_new = (w_t * v_t).astype(jnp.float32) - erase                   # (B,H,Dv)
        h = h + jnp.einsum("bhd,bhv->bhdv", k_t.astype(jnp.float32), v_new)
        o_t = jnp.einsum("bhdv,bhd->bhv", h, (q_t * scale).astype(jnp.float32))
        return h, o_t

    # move L to leading axis for scan
    to_scan = tuple(
        jnp.moveaxis(x, 1, 0) for x in (q, k, v, alpha, b, w)
    )
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

    gc = jnp.cumsum(g_raw_c.astype(f32), axis=1)  # chunk-local cumsum, inclusive -- (B,C,H,D)

    # decay_diff[b,h,i,j,d] = gc[b,i,h,d] - gc[b,j,h,d]
    gc_bhcd = jnp.moveaxis(gc, 2, 1)  # (B,H,C,D)
    decay_diff = gc_bhcd[:, :, :, None, :] - gc_bhcd[:, :, None, :, :]  # (B,H,C,C,D)
    edecay = jnp.exp(decay_diff)

    causal = jnp.tril(jnp.ones((C, C), dtype=f32))          # j<=i
    strict = jnp.tril(jnp.ones((C, C), dtype=f32), k=-1)    # j<i

    q_bhcd = jnp.moveaxis(q_c, 2, 1).astype(f32)  # (B,H,C,D)
    k_bhcd = jnp.moveaxis(k_c, 2, 1).astype(f32)
    b_bhcd = jnp.moveaxis(b_c, 2, 1).astype(f32)

    Aqk = scale * jnp.einsum("bhid,bhijd,bhjd->bhij", q_bhcd, edecay, k_bhcd) * causal
    bk_bhcd = b_bhcd * k_bhcd
    Akk = jnp.einsum("bhid,bhijd,bhjd->bhij", bk_bhcd, edecay, k_bhcd) * strict  # (B,H,C,C)

    eye = jnp.eye(C, dtype=f32)
    A = jnp.linalg.inv(eye[None, None] + Akk)  # (B,H,C,C) -- strictly lower Akk => always invertible

    kb_decayed = (b_c.astype(f32) * k_c.astype(f32)) * jnp.exp(gc)  # (B,C,H,D)
    w_pseudo = jnp.einsum("bhij,bjhd->bihd", A, kb_decayed)          # (B,C,H,D)
    u = jnp.einsum("bhij,bjhv->bihv", A, (w_c * v_c).astype(f32))    # (B,C,H,Dv)

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
        wh = jnp.einsum("bihd,bhdv->bihv", w_pseudo, h_pre)  # (B,C,H,Dv)
        v_new = u - wh

        # output: inter-chunk (q decayed @ h_pre) + intra-chunk (Aqk @ v_new)
        qh = jnp.einsum("bihd,bhdv->bihv", qg, h_pre)  # (B,C,H,Dv)
        v_new_bhcv = jnp.moveaxis(v_new, 2, 1)  # (B,H,C,Dv)
        intra = jnp.einsum("bhij,bhjv->bhiv", Aqk, v_new_bhcv)
        intra = jnp.moveaxis(intra, 1, 2)  # (B,C,H,Dv)
        o_c = (scale * qh + intra).astype(dtype)

        # inter-chunk state update: h_pre <- h_pre * exp(gc_last) + kg^T @ v_new
        decay_h = jnp.exp(gc_last)[..., None]  # (B,H,D,1)
        write = jnp.einsum("bihd,bihv->bhdv", kg, v_new)  # (B,H,D,Dv)
        h_new = h_pre * decay_h + write

        return h_new, o_c

    h_final, o_scanned = jax.lax.scan(
        chunk_step, h0, (q_ch, k_ch, v_ch, g_ch, b_ch, w_ch)
    )
    # o_scanned: (n_chunks, B, C, H, Dv) -> (B, L, H, Dv)
    o = jnp.moveaxis(o_scanned, 0, 1).reshape(bsz, L, H, Dv)
    return o, h_final
