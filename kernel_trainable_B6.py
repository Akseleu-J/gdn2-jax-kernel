"""
Milestone B6 (BACKWARD_PLAN.md) -- orchestration. Replaces the current
`_gdn2_core_bwd` in kernel_trainable.py (which computes the gradient via
`jax.vjp(gdn2_chunked_wy_reference, ...)` -- the "cheat" backward, re-tracing
the WHOLE forward through ordinary XLA autodiff every step) with the honest,
fused-Pallas backward chain: B2 -> B1 -> B3 -> B4 -> B5, exactly as
end-to-end validated in the B1-B5 integration test (see chat -- matched
gdn2_chunked_wy_reference's own jax.vjp gradient to ~1e-7 across 3 configs,
real TPU run).

Forward is UNCHANGED (still the already-validated Kernel A->B->C->D
pipeline, gdn2_pallas_forward) -- only `_gdn2_core_bwd` is replaced. The
`custom_vjp` scaffolding (`_gdn2_core`, `_gdn2_core_fwd`,
`gdn2_pallas_forward_trainable`) is identical in shape to the current
kernel_trainable.py; this file is meant to REPLACE that file's contents
once the double validation below (against gdn2_token_serial_reference AND
against the current cheat-backward) both pass on your v5e-8.

Chain (mirrors the already-validated integration test in
test_kernel_bwd_chain_full.py, just wrapped as the custom_vjp bwd rule):

    Kernel A/B/C forward (unchanged) -> saves Aqk, Akk, A, w_pseudo, u, kg,
      qg, gc_last as usual; ADDITIONALLY needs h_pre_all, v_new_all from
      Kernel D, which the standard gdn2_pallas_forward does NOT expose (see
      kernel_d_pipeline_PATCH.py) -- so the fwd residual-recompute inside
      this custom_vjp's bwd rule uses gdn2_inter_chunk_combine_with_state,
      not the production gdn2_inter_chunk_combine (same pattern the current
      _gdn2_core_bwd already uses: re-trace a slightly different variant of
      forward specifically for backward's needs).
    B2  (dav_backward_pallas)         -> dAqk, dv_partial
    B1  (gdn2_dhu_backward)           -> dh_all, dh0, dv_all
          dh_next_all = shift(dh_all) + dht
    B3  (wy_dqkg_backward_pallas)     -> dq1, dk1, dv, db1, dw, dgc1, dAkk
    B4  (intra_backward_pallas)       -> dq2, dk2, db2, dgc2   (using dAqk from B2, dAkk from B3)
    combine: dq=dq1+dq2, dk=dk1+dk2, db=db1+db2, dgc=dgc1+dgc2
    B5  (reverse_cumsum_pallas)       -> dg_raw

Returns (dq, dk, dv, dw, db, dg_raw, dh0) -- SAME order as the current
_gdn2_core_bwd's return tuple (matches _gdn2_core's residuals order:
q, k, v, w, b, g, h0).

NOT YET TESTED ON REAL TPU as the actual custom_vjp bwd rule (the individual
pieces have all been validated separately and as an unwired chain -- this
file's job is purely the wiring/plumbing, which is comparatively low risk,
but "low risk" is not "tested", per this project's own discipline -- run
test_kernel_trainable_b6.py before relying on this in real training).
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from kernel_a_scores import BT, build_chunk_scores_pallas
from kernel_b_solve import wy_solve_pallas
from kernel_c_recompute import recompute_wy_pallas
from kernel_d_pipeline import gdn2_pallas_forward

try:
    from kernel_d_pipeline import gdn2_inter_chunk_combine_with_state
except ImportError:
    from kernel_d_pipeline_PATCH import gdn2_inter_chunk_combine_with_state

from kernel_bwd_b1_dhu import gdn2_dhu_backward
from kernel_bwd_b2_dav import dav_backward_pallas
from kernel_bwd_b3_wy_dqkg import wy_dqkg_backward_pallas
from kernel_bwd_b4_intra import intra_backward_pallas
from kernel_bwd_b5_reverse_cumsum import reverse_cumsum_bwd

_HIGHEST = jax.lax.Precision.HIGHEST


def _reshape_in(t, bsz, n_chunks, H, D):
    """(B,L,H,D) -> (B,H,n_chunks,BT,D). Same helper duplicated in
    kernel_a_scores.py/kernel_c_recompute.py -- kept as a local closure
    here too, matching this project's existing convention (not a shared
    utility module anywhere yet)."""
    t = t.reshape(bsz, n_chunks, BT, H, D)
    return jnp.moveaxis(t, (1, 3), (2, 1))


def _reshape_out(t):
    """(B,H,n_chunks,BT,D) -> (B,L,H,D). Exact inverse of _reshape_in."""
    bsz, H, n_chunks, _BT, D = t.shape
    t2 = jnp.moveaxis(t, (1, 2, 3), (3, 1, 2))
    return t2.reshape(bsz, n_chunks * BT, H, D)


def _build_dh_next_all(dh_all, dht):
    """dh_next_all[:,:,c] = dh_all[:,:,c+1] for c<n_chunks-1, else dht --
    see kernel_bwd_b1_dhu.py's docstring for why this shift is needed (B3
    needs "the dh_carry that was live INSIDE B1's processing of this chunk",
    not dh_all[c] itself)."""
    shifted = dh_all[:, :, 1:]
    dht_expanded = dht[:, :, None]
    return jnp.concatenate([shifted, dht_expanded], axis=2)


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

    bsz, L, H, D = q.shape
    n_chunks = L // BT

    # --- forward residual-recompute (Kernel A/B/C unchanged; Kernel D's
    # "with_state" variant, per kernel_d_pipeline_PATCH.py, to expose
    # h_pre_all/v_new_all -- same pattern the CURRENT cheat-backward uses:
    # re-trace a forward variant tailored to backward's needs rather than
    # threading extra residuals through custom_vjp's fwd rule) ---
    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
    A = wy_solve_pallas(Akk)
    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
    o_all, h_final, h_pre_all, v_new_all = gdn2_inter_chunk_combine_with_state(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0
    )
    h_pre_all = jnp.moveaxis(h_pre_all, 0, 2)   # (n_chunks,B,H,D,D) -> (B,H,n_chunks,D,D)
    v_new_all = jnp.moveaxis(v_new_all, 0, 2)   # (n_chunks,B,H,BT,D) -> (B,H,n_chunks,BT,D)

    # gc (full, not just gc_last) -- recomputed the same tril-ones-matmul
    # trick Kernel A/C use internally; no forward kernel persists it.
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

    # --- backward chain: B2 -> B1 -> B3 -> B4 -> B5 ---
    dAqk, dv_partial = dav_backward_pallas(Aqk, v_new_all, do_r)

    dh_all, dh0, dv_all = gdn2_dhu_backward(
        do_r, dv_partial, w_pseudo, qg, kg, gc_last, scale, dht=dh_final
    )
    dh_next_all = _build_dh_next_all(dh_all, dh_final)

    b3_out = wy_dqkg_backward_pallas(
        q_r, k_r, b_r, w_r, v_r, gc, A, Akk, h_pre_all, v_new_all,
        do_r, dv_all, dh_next_all, scale,
    )

    dq4, dk4, db4, dgc4 = intra_backward_pallas(dAqk, b3_out["dAkk"], q, k, b, g, scale)

    dgc_total = b3_out["dgc"] + dgc4
    dg_raw = reverse_cumsum_bwd(dgc_total, chunk_size=BT)

    dq = _reshape_out(b3_out["dq"] + dq4)
    dk = _reshape_out(b3_out["dk"] + dk4)
    db = _reshape_out(b3_out["db"] + db4)
    dw = _reshape_out(b3_out["dw"])
    dv = _reshape_out(b3_out["dv_raw"])
    dg = _reshape_out(dg_raw)

    # Return order MUST match _gdn2_core's non-nondiff argument order:
    # q, k, v, w, b, g, h0.
    return dq, dk, dv, dw, db, dg, dh0


_gdn2_core.defvjp(_gdn2_core_fwd, _gdn2_core_bwd)


def gdn2_pallas_forward_trainable(q, k, v, w, b, g, scale, h0=None):
    """Drop-in trainable version of gdn2_pallas_forward -- differentiable
    w.r.t. q, k, v, w, b, g, h0 (scale is a static float, not differentiated).
    Identical signature/behavior to the current kernel_trainable.py version;
    only the backward implementation changed (fused Pallas chain instead of
    jax.vjp on gdn2_chunked_wy_reference).
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)
    return _gdn2_core(q, k, v, w, b, g, scale, h0)
