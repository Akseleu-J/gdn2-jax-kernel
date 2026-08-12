"""
Milestone B2 (BACKWARD_PLAN.md) -- Pallas TPU kernel for the intra-output
backward: given the upstream cotangent `do` (gradient w.r.t. o), the causal
score matrix `Aqk`, and `v_new` (both already computed/saved by the forward
pass), produce dAqk and the LOCAL (direct) contribution to dv_new.

Derivation (not transcribed from chunk_kda_bwd_kernel_dAv's transposed
block-pointer Triton code -- that's easy to misread; instead derived by hand
from our OWN already-validated forward formula and cross-checked against
jax.vjp, see verify_dav_backward() below):

    forward:  intra[i,:] = sum_j Aqk[i,j] * v_new[j,:]     (Aqk causal, i>=j)
    =>        dAqk[i,j]  = sum_v do[i,v] * v_new[j,v]        = (do @ v_newT)[i,j]
    =>        dv_new[j,:] = sum_i Aqk[i,j] * do[i,:]          = (AqkT @ do)[j,:]

Both masked to the same causal region Aqk itself occupies (i>=j) -- entries
outside that region are structurally zero in Aqk already, masking dAqk there
too keeps it clean for the next stage (Milestone B4) which backprops dAqk
through the Aqk-build formula.

NOTE: `dv_new` here is only the LOCAL contribution from the `Aqk @ v_new`
term. v_new also feeds the state-update path (h_new = h_pre*decay + kg^T@v_new),
handled separately by Milestone B1 -- the two contributions get SUMMED
before Milestone B3 uses the total dv_new.

NOT YET TESTED ON REAL TPU.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from kernel_a_scores import BT

_HIGHEST = jax.lax.Precision.HIGHEST


def _kernel_b2_body(aqk_ref, vnew_ref, do_ref, daqk_ref, dvnew_ref):
    Aqk = aqk_ref[0, 0, 0].astype(jnp.float32)     # (BT, BT)
    v_new = vnew_ref[0, 0, 0].astype(jnp.float32)  # (BT, D)
    do = do_ref[0, 0, 0].astype(jnp.float32)       # (BT, D)

    idx = jnp.arange(BT)
    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)  # (BT,BT), i>=j

    dAqk = jnp.dot(do, v_new.T, precision=_HIGHEST) * causal          # (BT,BT)
    dv_new = jnp.dot(Aqk.T, do, precision=_HIGHEST)                     # (BT,D)

    daqk_ref[0, 0, 0] = jnp.nan_to_num(dAqk, nan=0.0, posinf=1e4, neginf=-1e4)
    dvnew_ref[0, 0, 0] = jnp.nan_to_num(dv_new, nan=0.0, posinf=1e4, neginf=-1e4)


def dav_backward_pallas(Aqk, v_new, do):
    """Aqk: (B,H,n_chunks,BT,BT). v_new, do: (B,H,n_chunks,BT,D).
    Returns dAqk: (B,H,n_chunks,BT,BT), dv_new_local: (B,H,n_chunks,BT,D).
    """
    bsz, H, n_chunks, _BT, D = v_new.shape
    grid = (bsz, H, n_chunks)

    aqk_spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))
    io_spec = pl.BlockSpec((1, 1, 1, BT, D), lambda i, h, c: (i, h, c, 0, 0))

    dAqk, dv_new = pl.pallas_call(
        _kernel_b2_body,
        grid=grid,
        in_specs=[aqk_spec, io_spec, io_spec],
        out_specs=[aqk_spec, io_spec],
        out_shape=[
            jax.ShapeDtypeStruct(Aqk.shape, jnp.float32),
            jax.ShapeDtypeStruct(v_new.shape, jnp.float32),
        ],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=64 * 1024 * 1024),
    )(Aqk, v_new, do)

    return dAqk, dv_new


# ==========================================================================
# Plain-JAX formula + jax.vjp cross-check (run this locally before trusting
# the Pallas kernel above -- same "derive, then verify with autodiff" pattern
# used throughout this project).
# ==========================================================================
def dav_backward_formula(Aqk, v_new, do):
    """Same formula as the Pallas kernel, in plain JAX, for the cross-check."""
    bsz, H, n_chunks, BTl, D = v_new.shape
    idx = jnp.arange(BTl)
    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    dAqk = jnp.einsum("bhcid,bhcjd->bhcij", do, v_new, precision=_HIGHEST) * causal
    dv_new = jnp.einsum("bhcij,bhcid->bhcjd", Aqk, do, precision=_HIGHEST)
    return dAqk, dv_new


def verify_dav_backward(key, B=1, H=2, n_chunks=2, D=16):
    """Sanity check: dav_backward_formula must match jax.vjp on the actual
    Aqk @ v_new einsum. Run this (plain JAX/numpy, no Pallas, no TPU needed)
    before trusting the Pallas kernel's math."""
    import numpy as np

    k1, k2, k3 = jax.random.split(key, 3)
    idx = jnp.arange(BT)
    causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
    Aqk = jax.random.normal(k1, (B, H, n_chunks, BT, BT)) * causal
    v_new = jax.random.normal(k2, (B, H, n_chunks, BT, D))
    do = jax.random.normal(k3, (B, H, n_chunks, BT, D))

    def intra_fn(Aqk_, v_new_):
        return jnp.einsum("bhcij,bhcjd->bhcid", Aqk_, v_new_, precision=_HIGHEST)

    _, vjp_fn = jax.vjp(intra_fn, Aqk, v_new)
    dAqk_ref, dv_new_ref = vjp_fn(do)

    dAqk_formula, dv_new_formula = dav_backward_formula(Aqk, v_new, do)

    dAqk_rel = float(jnp.max(jnp.abs(dAqk_ref - dAqk_formula)) / (jnp.max(jnp.abs(dAqk_ref)) + 1e-12))
    dv_new_rel = float(jnp.max(jnp.abs(dv_new_ref - dv_new_formula)) / (jnp.max(jnp.abs(dv_new_ref)) + 1e-12))
    return dAqk_rel, dv_new_rel


if __name__ == "__main__":
    # Pure-JAX/numpy check -- run this first, works without TPU.
    rel_errs = verify_dav_backward(jax.random.PRNGKey(0))
    print(f"dAqk_rel={rel_errs[0]:.3e}  dv_new_rel={rel_errs[1]:.3e}")
    assert rel_errs[0] < 1e-5 and rel_errs[1] < 1e-5, "formula does not match jax.vjp -- do not trust the Pallas kernel yet"
    print("Formula matches jax.vjp -- Pallas kernel (dav_backward_pallas) math should be correct, test on TPU next.")
