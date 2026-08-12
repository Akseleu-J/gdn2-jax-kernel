"""
Milestone 3 -- Kernel A (Pallas/TPU): builds the intra-chunk Aqk/Akk score
matrices for one (batch, head, chunk). This is the FIRST Pallas kernel of the
staged port (per agreed architecture: separate kernel per stage, fp32 solve,
fixed seq_len, n_heads=6 -> d_head=128 exactly matching the v5e MXU tile).

NOT YET TESTED ON REAL TPU -- the sandbox that produced this file has no
JAX/TPU access. Please run test_kernel_a.py on your v5e-8 and report back
errors/mismatches so we can iterate, same as we did for Milestone 1.

Numerical strategy (deliberate choice, not the original authors' fastest
path): builds each BCxBC sub-block of Aqk/Akk via the DIRECT per-pair decay
difference (exp(gc_i - gc_j), always <=1 for causal i>=j since decay<=1 --
unconditionally safe), matching Kernel 1's approach in chunk_gdn2.py,
generalized from "diagonal sub-block only" to "every sub-block pair in the
lower triangle" (2 diagonal + 1 off-diagonal for BC=128, BT=256). The
original authors' Kernel 2/3 use a faster factored-matmul trick with local
decay re-centering for the off-diagonal blocks -- that is a real optimization
opportunity for later (after correctness is confirmed), not implemented here
to avoid porting a numerically trickier scheme before the simple one is
proven correct on your hardware.

Shapes: q, k, b (erase gate), g (raw per-token log-decay): (B, L, H, D).
Output: Aqk, Akk: (B, H, n_chunks, BT, BT), float32.
D must be 128 (n_heads=6 for d_model=768 gives d_head=128 exactly -- update
ModelConfig.n_heads=6 before wiring this in).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_HIGHEST = jax.lax.Precision.HIGHEST

BT = 256   # chunk size (matches cfg.deltanet_chunk_size elsewhere in the codebase)
BC = 128   # sub-chunk size == MXU tile
N_SUB = BT // BC  # 2


def _weighted_pair_sum(a_i, edecay, b_j):
    """Computes sum_d a_i[i,d] * edecay[i,j,d] * b_j[j,d] -> (BC,BC), via
    explicit broadcast-multiply + reduce instead of a 3-operand einsum.

    jnp.einsum("id,ijd,jd->ij", ...) lowers to a batched dot_general on TPU
    whose dimension-number encoding Mosaic in this JAX version fails to parse
    (MLIRError: failed to parse TPU_DotDimensionNumbersAttr). Plain
    elementwise multiply + jnp.sum avoids dot_general entirely and only uses
    ops (mul, reduce-sum over the last axis) that Mosaic lowers natively.
    """
    tmp = a_i[:, None, :] * edecay      # (BC, BC, D)
    tmp = tmp * b_j[None, :, :]          # (BC, BC, D)
    return jnp.sum(tmp, axis=-1)           # (BC, BC)


def _kernel_a_body(q_ref, k_ref, b_ref, g_ref, aqk_ref, akk_ref, *, scale):
    # Each ref has block shape (1, 1, 1, BT, D) / (1, 1, 1, BT, BT) -- index
    # away the three singleton (batch, head, chunk) dims explicitly, Pallas
    # refs don't support .reshape() the way plain arrays do.
    q_full = q_ref[0, 0, 0].astype(jnp.float32)   # (BT, D)
    k_full = k_ref[0, 0, 0].astype(jnp.float32)
    b_full = b_ref[0, 0, 0].astype(jnp.float32)
    g_raw = g_ref[0, 0, 0].astype(jnp.float32)

    bt_idx = jnp.arange(BT)
    tril_ones_bt = (bt_idx[:, None] >= bt_idx[None, :]).astype(jnp.float32)  # (BT,BT) inclusive lower-tri
    gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)  # chunk-local cumulative log-decay
    # NOTE: cumsum isn't lowered by Mosaic/Pallas-TPU ("Unimplemented primitive
    # ... cumsum") -- inclusive prefix-sum via a lower-triangular ones-matmul
    # is the standard workaround and gives the identical result:
    #   gc[i,d] = sum_{j<=i} g_raw[j,d]

    # zero-init full outputs (upper-triangular sub-blocks stay zero)
    aqk_ref[0, 0, 0] = jnp.zeros((BT, BT), dtype=jnp.float32)
    akk_ref[0, 0, 0] = jnp.zeros((BT, BT), dtype=jnp.float32)

    for si in range(N_SUB):
        for sj in range(si + 1):  # lower triangle only (si >= sj)
            i0, i1 = si * BC, (si + 1) * BC
            j0, j1 = sj * BC, (sj + 1) * BC

            q_i = q_full[i0:i1]      # (BC, D)
            k_i = k_full[i0:i1]
            k_j = k_full[j0:j1]
            b_i = b_full[i0:i1]
            gc_i = gc[i0:i1]
            gc_j = gc[j0:j1]

            decay_diff = gc_i[:, None, :] - gc_j[None, :, :]  # (BC, BC, D)
            # ФИКС (пользователь, до инцидента 710+): тот же клип, что в
            # gdn2_wy_reference.py -- без него падало немедленно, не через
            # 700 шагов. nan_to_num на aqk_blk/akk_blk ниже -- вторая линия.
            edecay = jnp.exp(jnp.clip(decay_diff, -20.0, 20.0))

            aqk_blk = scale * _weighted_pair_sum(q_i, edecay, k_j)
            bk_i = b_i * k_i
            akk_blk = _weighted_pair_sum(bk_i, edecay, k_j)

            if si == sj:
                idx = jnp.arange(BC)
                causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)  # j<=i
                strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)   # j<i
                aqk_blk = aqk_blk * causal
                akk_blk = akk_blk * strict
            # off-diagonal (si>sj): every pair already satisfies j<i globally,
            # no masking needed.

            aqk_ref[0, 0, 0, i0:i1, j0:j1] = jnp.nan_to_num(aqk_blk, nan=0.0, posinf=1e4, neginf=-1e4)
            akk_ref[0, 0, 0, i0:i1, j0:j1] = jnp.nan_to_num(akk_blk, nan=0.0, posinf=1e4, neginf=-1e4)


def build_chunk_scores_pallas(q, k, b, g, scale):
    """q, k, b, g: (B, L, H, D), D must equal 128. Returns Aqk, Akk: (B, H, n_chunks, BT, BT)."""
    bsz, L, H, D = q.shape
    assert D == 128, f"Kernel A assumes d_head=128 (MXU tile); got D={D}."
    assert L % BT == 0, f"seq_len={L} must be divisible by BT={BT}."
    n_chunks = L // BT

    # (B, L, H, D) -> (B, H, n_chunks, BT, D) so the grid can block cleanly
    def reshape_in(t):
        t = t.reshape(bsz, n_chunks, BT, H, D)
        return jnp.moveaxis(t, (1, 3), (2, 1))  # -> (B, H, n_chunks, BT, D)

    q_r, k_r, b_r, g_r = map(reshape_in, (q, k, b, g))

    grid = (bsz, H, n_chunks)

    in_spec = pl.BlockSpec(
        (1, 1, 1, BT, D), lambda i, h, c: (i, h, c, 0, 0)
    )
    out_spec = pl.BlockSpec(
        (1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0)
    )

    aqk, akk = pl.pallas_call(
        lambda *refs: _kernel_a_body(*refs, scale=scale),
        grid=grid,
        in_specs=[in_spec, in_spec, in_spec, in_spec],
        out_specs=[out_spec, out_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, BT), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, BT), jnp.float32),
        ],
        # Default scoped-VMEM limit is 16MB; the (BC,BC,D)=128x128x128 fp32
        # intermediates in _weighted_pair_sum (~8.4MB each, several alive per
        # sub-block) need more headroom. 100MiB leaves ~28MiB of the v5e's
        # 128MiB/core budget for the input/output double-buffering Pallas
        # sets up automatically around the grid.
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 * 1024),
    )(q_r, k_r, b_r, g_r)

    return aqk, akk
