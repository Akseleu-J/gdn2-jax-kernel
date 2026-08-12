"""
Milestone 6 -- trainable wrapper: Pallas forward (fast, Milestone 3) +
custom_vjp backward computed via jax.vjp on the pure-JAX WY-chunked reference
(gdn2_chunked_wy_reference, Milestone 1 -- already validated as
mathematically identical to the Pallas pipeline, see test_gdn2_pipeline_e2e.py).

Rationale: pallas_call does not support automatic reverse-mode
differentiation in this JAX version ("Linearization failed to produce known
values for all output primals", confirmed on real v5e-8 -- fails on the very
first Pallas kernel, Kernel A). Deriving an exact analytical backward pass
for the WY-solve by hand is substantial, error-prone work. Instead: forward
pass uses the fast Pallas kernels; backward pass re-traces the equivalent
pure-JAX implementation (ordinary XLA, no Mosaic involved, so ordinary
autodiff just works) to get exact gradients. Optimizing backward into its
own fused Pallas kernel is a later step (see the original Milestone 6
backlog item), not required for correctness.

NOT YET TESTED ON REAL TPU.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from atomic_ops.kernel_a_scores import BT
from atomic_ops.kernel_d_pipeline import gdn2_pallas_forward
from atomic_ops.gdn2_wy_reference import gdn2_chunked_wy_reference

@partial(jax.custom_vjp, nondiff_argnums=(6,))
def _gdn2_core(q, k, v, w, b, g, scale, h0):
    return gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=h0)


def _gdn2_core_fwd(q, k, v, w, b, g, scale, h0):
    out = gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=h0)
    residuals = (q, k, v, w, b, g, h0)
    return out, residuals


def _gdn2_core_bwd(scale, residuals, cotangents):
    q, k, v, w, b, g, h0 = residuals
    do, dh_final = cotangents

    def ref_forward(q_, k_, v_, w_, b_, g_, h0_):
        return gdn2_chunked_wy_reference(q_, k_, v_, g_, b_, w_, scale, chunk_size=BT, h0=h0_)

    _, vjp_fn = jax.vjp(ref_forward, q, k, v, w, b, g, h0)
    dq, dk, dv, dw, db, dg, dh0 = vjp_fn((do, dh_final))
    return dq, dk, dv, dw, db, dg, dh0


_gdn2_core.defvjp(_gdn2_core_fwd, _gdn2_core_bwd)


def gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale, h0=None):
    """Drop-in trainable version of gdn2_pallas_forward -- differentiable
    w.r.t. q, k, v, w, b, g, h0 (scale is a static float, not differentiated).
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    return _gdn2_core(q, k, v, w, b, g, scale, h0)
