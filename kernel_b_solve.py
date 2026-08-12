"""
Milestone 3 -- Kernel B (Pallas/TPU): WY solve A = (I + Akk)^{-1}.

Lessons carried over from Kernel A's debugging on real v5e-8 hardware:
  - avoid multi-operand einsum -> Mosaic's dot_general dimension-number
    parser failed on the batched contraction it produced. Row-vector times
    matrix here is done via explicit broadcast-multiply + jnp.sum instead.
  - plain 2D jnp.dot (no batch dims) DID work fine (used for the cumsum
    replacement in Kernel A) -- used again below for the off-diagonal block
    assembly (T10 @ A00, A11 @ (...)).
  - writing outputs via direct ref-slice assignment (out_ref[0,0,0,i0:i1,...] = ...)
    worked reliably in Kernel A -- used again here instead of jnp.concatenate,
    which hasn't been verified to lower cleanly in this Mosaic version.
  - set vmem_limit_bytes explicitly from the start (default scoped-VMEM limit
    is 16MB and Kernel A OOM'd against it).

Math: Akk (BT,BT) is strictly lower triangular. With BT=2*BC (N_SUB=2, the
config we locked in: BT=256, BC=128), split into 2x2 blocks:
    Akk = [[T00,  0 ],
           [T10, T11]]
(I+Akk) is block lower-triangular too, so its inverse is:
    A00 = (I+T00)^{-1}
    A11 = (I+T11)^{-1}
    A10 = -A11 @ T10 @ A00
Each A00/A11 is a BC x BC strictly-lower-triangular solve, done via row-by-row
forward substitution (jax.lax.fori_loop) -- this is the same algebra as
_wy_inverse() in gdn2_wy_reference.py, just scoped to one BCxBC sub-block
instead of the full BT for a shorter serial loop (128 steps instead of 256),
matching the block design agreed in Milestone 2.

NOT YET TESTED ON REAL TPU.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from kernel_a_scores import BT, BC, N_SUB

_HIGHEST = jax.lax.Precision.HIGHEST

assert N_SUB == 2, "Kernel B currently implements only the 2-subblock (BT=2*BC) case."


def _bc_forward_substitution(T):
    """A = (I+T)^{-1} for strictly lower-triangular T: (BC, BC) -> (BC, BC).

    Row-by-row forward substitution: A[i,:] = e_i - T[i,:] @ A (using only
    rows < i, which is all that's nonzero in T[i,:] since T is strictly
    lower triangular). Both the row-extraction (T[i,:]) and the row-update
    (A[i,:] <- new_row) are done via one-hot mask + multiply/sum instead of
    jax.lax.dynamic_slice(_in_dim)/dynamic_update_slice(_in_dim) -- Mosaic
    does not lower the dynamic_slice primitive on plain values inside a
    fori_loop body ("Unimplemented primitive ... dynamic_slice", found on
    real v5e-8 hardware). Masked multiply+sum only uses ops already proven
    to lower fine (same pattern as Kernel A's _weighted_pair_sum).
    """
    bc = T.shape[-1]
    idx = jnp.arange(bc)

    def body(i, A):
        onehot_i = (idx == i).astype(jnp.float32)              # (bc,), 1 at position i
        t_row = jnp.sum(T * onehot_i[:, None], axis=0)           # (bc,) == T[i, :]
        contrib = jnp.sum(t_row[:, None] * A, axis=0)              # (bc,) == T[i,:] @ A
        new_row = onehot_i - contrib                                  # (bc,)
        mask_col = onehot_i[:, None]                                    # (bc,1), 1 only at row i
        A = A * (1.0 - mask_col) + mask_col * new_row[None, :]
        return A

    A0 = jnp.zeros((bc, bc), dtype=jnp.float32)
    return jax.lax.fori_loop(0, bc, body, A0)


def _kernel_b_body(akk_ref, a_ref):
    Akk = akk_ref[0, 0, 0].astype(jnp.float32)  # (BT, BT)

    T00 = Akk[0:BC, 0:BC]
    T11 = Akk[BC:2 * BC, BC:2 * BC]
    T10 = Akk[BC:2 * BC, 0:BC]

    A00 = _bc_forward_substitution(T00)
    A11 = _bc_forward_substitution(T11)

    tmp = jnp.dot(T10, A00, precision=_HIGHEST)     # (BC, BC) -- plain 2D dot, no batch dims
    A10 = -jnp.dot(A11, tmp, precision=_HIGHEST)      # (BC, BC)

    # ФИКС: тот же рубеж защиты, что добавлен в gdn2_wy_reference.py после
    # инцидента на реальном обучении (шаг 710+, non-finite delta в gdn2) --
    # Akk/A могут дрейфовать в нестабильный режим по мере обучения, forward
    # Pallas-путь нуждается в той же санитизации, что и backward-референс.
    A00 = jnp.nan_to_num(A00, nan=0.0, posinf=1e4, neginf=-1e4)
    A10 = jnp.nan_to_num(A10, nan=0.0, posinf=1e4, neginf=-1e4)
    A11 = jnp.nan_to_num(A11, nan=0.0, posinf=1e4, neginf=-1e4)

    a_ref[0, 0, 0] = jnp.zeros((BT, BT), dtype=jnp.float32)
    a_ref[0, 0, 0, 0:BC, 0:BC] = A00
    a_ref[0, 0, 0, BC:2 * BC, 0:BC] = A10
    a_ref[0, 0, 0, BC:2 * BC, BC:2 * BC] = A11


def wy_solve_pallas(Akk):
    """Akk: (B, H, n_chunks, BT, BT) strictly lower triangular per chunk.
    Returns A = (I + Akk)^{-1}: same shape, float32.
    """
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
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
    )(Akk)

    return A
