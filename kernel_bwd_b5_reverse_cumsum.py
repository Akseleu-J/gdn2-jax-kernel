"""
Milestone B5 (BACKWARD_PLAN.md) -- reverse cumsum for dg.

Forward (Kernel A / Kernel C, both use the same trick):
    gc[i,d] = sum_{j<=i} g_raw[j,d]      via gc = tril_ones @ g_raw

This is a linear map g_raw -> gc with matrix M = tril_ones (inclusive lower
triangular). Its VJP is multiplication by M^T:
    dg_raw = M^T @ dgc = triu_ones @ dgc      where triu_ones[i,j] = 1 if j>=i else 0
           = tril_ones.T @ dgc

i.e. dg_raw[i,d] = sum_{j>=i} dgc[j,d]  -- a REVERSE (suffix) cumulative sum.
Trivial to verify: for M=tril_ones, (M@x)^T@y = x^T@(M^T@y) for any x,y --
standard adjoint-of-a-linear-map identity, not something that needs
empirical cross-checking the way the more complex B1/B2/B3/B4 pieces did.

Same Mosaic limitation as forward's cumsum (see kernel_a_scores.py's own
note): jnp.cumsum (and presumably jnp.cumsum(..., reverse=True)) isn't a
lowered TPU Pallas primitive, so this uses the same triangular-ones-matmul
trick, transposed.

NOT YET TESTED ON REAL TPU (though this is about as low-risk as B1 -- a
single matmul, same pattern already proven to lower fine in Kernel A/C).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST


def reverse_cumsum_bwd(dgc, chunk_size):
    """dgc: (..., C, D) gradient w.r.t. the chunk-local cumulative decay gc.
    Returns dg_raw: (..., C, D) gradient w.r.t. the raw per-token log-decay
    g_raw, via the suffix-sum adjoint of the forward's tril_ones@g_raw.

    chunk_size (=C) is passed explicitly (not inferred from dgc.shape[-2])
    so this can be used identically whether dgc is a single chunk (C=BT) or
    already batched over chunks with C as the second-to-last axis.
    """
    C = chunk_size
    idx = jnp.arange(C)
    triu_ones = (idx[:, None] <= idx[None, :]).astype(jnp.float32)  # (C,C), triu_ones[i,j]=1 if j>=i

    # dgc: (..., C, D) -> contract over the C axis (second-to-last) with triu_ones
    # dg_raw[..., i, d] = sum_j triu_ones[i,j] * dgc[..., j, d]
    dg_raw = jnp.einsum("ij,...jd->...id", triu_ones, dgc.astype(jnp.float32), precision=_HIGHEST)
    return dg_raw


def verify_reverse_cumsum(key, C=16, D=8, batch_shape=(2, 3)):
    """Pure-JAX/numpy sanity check (no TPU needed): reverse_cumsum_bwd must
    equal jax.vjp on the forward tril_ones@g_raw cumsum trick."""
    import numpy as np

    k1, k2 = jax.random.split(key)
    g_raw = jax.random.normal(k1, batch_shape + (C, D))
    dgc = jax.random.normal(k2, batch_shape + (C, D))

    idx = jnp.arange(C)
    tril_ones = (idx[:, None] >= idx[None, :]).astype(jnp.float32)

    def cumsum_fwd(g_raw_):
        return jnp.einsum("ij,...jd->...id", tril_ones, g_raw_, precision=_HIGHEST)

    _, vjp_fn = jax.vjp(cumsum_fwd, g_raw)
    (dg_raw_ref,) = vjp_fn(dgc)

    dg_raw_formula = reverse_cumsum_bwd(dgc, chunk_size=C)

    rel = float(jnp.max(jnp.abs(dg_raw_ref - dg_raw_formula)) / (jnp.max(jnp.abs(dg_raw_ref)) + 1e-12))
    return rel


if __name__ == "__main__":
    rel = verify_reverse_cumsum(jax.random.PRNGKey(20))
    print(f"reverse_cumsum rel_err={rel:.3e}")
    assert rel < 1e-5, "reverse cumsum does not match jax.vjp"
    print("B5 formula matches jax.vjp. Run this on TPU too (Mosaic lowering of the matmul, same as Kernel A's).")
