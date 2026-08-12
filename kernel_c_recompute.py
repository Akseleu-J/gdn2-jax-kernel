"""
Milestone 3 -- Kernel C (Pallas/TPU): recompute_w_u_fwd_gdn2_kernel equivalent.
Builds w_pseudo, u, kg, qg, gc_last from A (Kernel B's WY-solve output) and
the original chunk inputs q, k, v, w, b, g.

Formulas (from recompute_w_u_fwd_gdn2_kernel, chunk_gdn2.py L698-790,
validated against the token-serial ground truth via Milestone 1):
    gc          = chunk-local cumulative log-decay (same tril-ones-matmul
                  trick as Kernel A -- cumsum isn't lowered by Mosaic)
    w_pseudo    = A @ (b * k * exp(gc))
    u           = A @ (w * v)
    kg          = k * exp(gc_last - gc)
    qg          = q * exp(gc)
    gc_last     = gc[-1, :]   (decay accumulated over the whole chunk)

All contractions here are plain 2-operand (BT,BT)@(BT,D) matmuls -- the same
shape pattern as the tril-ones cumsum trick that already worked in Kernel A,
so jnp.dot is used directly (no einsum, no dynamic_slice/dynamic_update_slice
-- lessons from Kernels A/B).

NOT YET TESTED ON REAL TPU.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from kernel_a_scores import BT

_HIGHEST = jax.lax.Precision.HIGHEST


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
    gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)  # (BT, D), chunk-local cumulative decay

    kb_decayed = b * k * jnp.exp(gc)                          # (BT, D)
    w_pseudo = jnp.dot(A, kb_decayed, precision=_HIGHEST)       # (BT, D)
    u = jnp.dot(A, w * v, precision=_HIGHEST)                     # (BT, D)
    # ФИКС: см. тот же комментарий в kernel_b_solve.py -- рубеж защиты после
    # инцидента на реальном обучении.
    w_pseudo = jnp.nan_to_num(w_pseudo, nan=0.0, posinf=1e4, neginf=-1e4)
    u = jnp.nan_to_num(u, nan=0.0, posinf=1e4, neginf=-1e4)

    gc_last_row = gc[BT - 1]  # (D,) -- static index (compile-time constant), not dynamic_slice
    kg = k * jnp.exp(gc_last_row[None, :] - gc)  # (BT, D)
    qg = q * jnp.exp(gc)                            # (BT, D)

    w_pseudo_ref[0, 0, 0] = w_pseudo
    u_ref[0, 0, 0] = u
    kg_ref[0, 0, 0] = kg
    qg_ref[0, 0, 0] = qg
    gc_last_ref[0, 0, 0, 0] = gc_last_row


def recompute_wy_pallas(q, k, v, w, b, g, A):
    """q,k,b,g,w,v: (B,L,H,D) (chunked internally, L must be divisible by BT).
    A: (B,H,n_chunks,BT,BT) -- from wy_solve_pallas.
    Returns w_pseudo, u, kg, qg: (B,H,n_chunks,BT,D); gc_last: (B,H,n_chunks,D).
    """
    bsz, L, H, D = q.shape
    assert L % BT == 0
    n_chunks = L // BT

    def reshape_in(t):
        t = t.reshape(bsz, n_chunks, BT, H, D)
        return jnp.moveaxis(t, (1, 3), (2, 1))  # -> (B, H, n_chunks, BT, D)

    q_r, k_r, v_r, w_r, b_r, g_r = map(reshape_in, (q, k, v, w, b, g))

    grid = (bsz, H, n_chunks)
    io_spec = pl.BlockSpec((1, 1, 1, BT, D), lambda i, h, c: (i, h, c, 0, 0))
    a_spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))
    # gc_last: (B,H,n_chunks,D) with block (1,1,1,D) failed Mosaic's blockspec
    # check -- the block's second-to-last dim (n_chunks=1) is neither
    # divisible by 8 nor equal to the full n_chunks (found on real v5e-8: "last
    # two dimensions of your block shape are divisible by 8 and 128 ... or be
    # equal to the respective dimensions of the overall array"). Adding a
    # dummy trailing size-1 axis makes the block's last two dims (1,D) match
    # the full array's last two dims (1,D) exactly, which is allowed -- then
    # squeeze it back off outside the pallas_call.
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

    gc_last = gc_last.reshape(bsz, H, n_chunks, D)  # drop the dummy axis added for the blockspec constraint
    return w_pseudo, u, kg, qg, gc_last
