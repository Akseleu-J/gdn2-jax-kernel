"""
Milestone B1 (BACKWARD_PLAN.md) -- backward through the inter-chunk state
recurrence. Reference: chunk_kda.py's
chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64 /
chunk_gated_delta_rule_bwd_dhu, but implemented here in plain JAX (per the
plan: "простая, некомпилируемая в Pallas часть ... можно сразу писать на
чистом JAX, не Pallas" -- same reasoning as forward Kernel D).

Forward recurrence being differentiated (kernel_d_pipeline.gdn2_inter_chunk_combine,
per-chunk c, entering state h_pre):
    wh    = w_pseudo_c @ h_pre
    v_new = u_c - wh
    qh    = qg_c @ h_pre
    intra = Aqk_c @ v_new
    o_c   = scale*qh + intra
    h_new = h_pre * exp(gc_last_c) + kg_c^T @ v_new

o_c is additively split into a qh-term (depends on h_pre only) and an
intra-term (depends on v_new only, via Aqk -- that's Milestone B2's job,
chunk_kda_bwd_dAv). This milestone owns everything that flows through h_pre:
the qh-term's contribution to dh_pre, the state-recurrence's contribution to
dh_pre (via decay) and to dv_new (via the kg^T@v_new write), AND -- easy to
miss, see note below -- h_pre's contribution via v_new = u - w_pseudo@h_pre
(v_new depends on h_pre too, not just on u/w_pseudo as free inputs). It takes
dv_partial (from B2) and adds this milestone's own contributions to produce
the TOTAL dv_new that Milestone B3 needs, and the TOTAL dh_pre gradient.

IMPORTANT CORRECTION (found via cross-check against jax.vjp on the ACTUAL
gdn2_inter_chunk_combine, not just an isolated toy recurrence -- see
test_kernel_bwd_b1.py): an earlier draft of this kernel omitted the
v_new = u - w_pseudo@h_pre pathway entirely, having only been validated
against a toy version where v_new was a free/independent input. That toy
version matched jax.vjp to float32 roundoff but was incomplete -- it's only
valid for the isolated qh/write terms, not the real forward. The plan's own
input list for this step (chunk_gated_delta_rule_bwd_dhu's signature) already
names `w_wy` (== w_pseudo) as a required input, which was the tell. Once
w_pseudo is threaded through and this third term added, the FULL
end-to-end check against jax.vjp on gdn2_inter_chunk_combine (via
gdn2_inter_chunk_combine_with_state) matches exactly (0.0 relative error in
float32) across all tested shape configs, including multi-chunk cases.

Derivation (by hand; cross-checked against jax.vjp end-to-end, see
test_kernel_bwd_b1.py and the CPU self-check run in chat -- exact match,
0.0 relative error in float32, across 4 shape configs incl. n_chunks up to 4):

  Running chunks in REVERSE (c = n_chunks-1 down to 0), carrying
  dh_carry = dL/dh_pre_{c+1} (init to dht = dL/dh_final for the last chunk):

    dqh           = scale * do_c
    dh_from_out   = qg_c^T @ dqh                       # (D,D), via qh = qg_c @ h_pre
    dh_from_state = dh_carry * exp(gc_last_c)             # (D,D), via h_new = h_pre*decay + ...
    dv_write      = kg_c @ dh_carry                          # (BT,D), via h_new = ... + kg_c^T@v_new
    dv_new_c      = dv_partial_c + dv_write                    # TOTAL dv_new for this chunk
    dh_from_vnew  = -w_pseudo_c^T @ dv_new_c                     # (D,D), via v_new = u - w_pseudo@h_pre
    dh_pre_c      = dh_from_out + dh_from_state + dh_from_vnew     # <- this chunk's incoming-state grad

  dh_carry becomes dh_pre_c for the next (earlier) iteration; after the last
  (c=0) iteration, dh_carry == dh0 = dL/dh0.

NOT YET TESTED ON REAL TPU -- validated on CPU/plain JAX only so far (no
Pallas involved in this milestone, so this is lower-risk than the Pallas
kernels, same reasoning as forward Kernel D getting less scrutiny than
Kernels A-C).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST


def gdn2_dhu_backward(do, dv_partial, w_pseudo, qg, kg, gc_last, scale, dht=None):
    """B1: reverse-chunk backward through the inter-chunk state recurrence.

    do:         (B,H,n_chunks,BT,D)  -- upstream cotangent on o (whole thing,
                                        both the qh-term and intra-term parts
                                        -- only the qh-term is used directly
                                        here; the intra-term's effect comes in
                                        via dv_partial).
    dv_partial: (B,H,n_chunks,BT,D)  -- from Milestone B2 (chunk_kda_bwd_dAv):
                                        the intra = Aqk@v_new path's
                                        contribution to dv_new.
    w_pseudo:   (B,H,n_chunks,BT,D)  -- Kernel C output ("w_wy" in the plan).
                                        REQUIRED: v_new = u - w_pseudo@h_pre,
                                        so h_pre's gradient also flows back
                                        through this term -- easy to miss,
                                        see the module docstring's "IMPORTANT
                                        CORRECTION" note.
    qg, kg:     (B,H,n_chunks,BT,D)  -- Kernel C outputs (already computed,
                                        forward-saved).
    gc_last:    (B,H,n_chunks,D)     -- Kernel C output (chunk-local
                                        cumulative log-decay, last row).
    scale:      python float, static (same scale used in forward).
    dht:        (B,H,D,D) or None    -- cotangent on h_final (zeros if this
                                        GDN-2 layer's state has no downstream
                                        consumer, e.g. no cross-segment state
                                        passing yet in this model).

    Returns:
      dh_all: (B,H,n_chunks,D,D)  -- dL/dh_pre_c for every chunk c (needed by
                                     Milestone B3's dq/w_pseudo-path gradients)
      dh0:    (B,H,D,D)            -- dL/dh0
      dv_all: (B,H,n_chunks,BT,D) -- TOTAL dv_new (dv_partial + this
                                     milestone's write-path contribution) --
                                     feeds Milestone B3.
    """
    bsz, H, n_chunks, BT, D = qg.shape
    if dht is None:
        dht = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (do, dv_partial, w_pseudo, qg, kg, gc_last))

    def step(dh_carry, inputs):
        do_c, dvp_c, wp_c, qg_c, kg_c, gclast_c = inputs
        decay_c = jnp.exp(gclast_c)[..., None]  # (B,H,D,1)

        dqh = scale * do_c
        contrib_from_output = jnp.einsum("bhid,bhiv->bhdv", qg_c, dqh, precision=_HIGHEST)
        contrib_from_state = dh_carry * decay_c

        dv_write = jnp.einsum("bhid,bhdv->bhiv", kg_c, dh_carry, precision=_HIGHEST)
        dv_new_c = dvp_c + dv_write
        dv_new_c = jnp.nan_to_num(jnp.clip(dv_new_c, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)

        # v_new_c = u_c - w_pseudo_c @ h_pre  =>  d(h_pre) -= w_pseudo_c^T @ dv_new_c
        contrib_from_vnew = -jnp.einsum("bhjd,bhjv->bhdv", wp_c, dv_new_c, precision=_HIGHEST)

        dh_pre_c = contrib_from_output + contrib_from_state + contrib_from_vnew
        # ФИКС (см. HANDOFF.md §6 / BACKWARD_PLAN.md incident): backward-only
        # sanitization, same pattern as the forward-side patches -- a
        # non-finite upstream cotangent (do/dht) or an exploded decay term
        # should not silently propagate NaN through the whole reverse scan.
        dh_pre_c = jnp.nan_to_num(jnp.clip(dh_pre_c, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)

        return dh_pre_c, (dh_pre_c, dv_new_c)

    dh0, (dh_all_rev, dv_all_rev) = jax.lax.scan(step, dht, to_scan, reverse=True)
    # NOTE: jax.lax.scan(reverse=True) still returns ys stacked in ORIGINAL
    # index order (0..n_chunks-1) even though the loop body executes from
    # the last chunk to the first -- no manual re-flip needed here.
    dh_all = jnp.moveaxis(dh_all_rev, 0, 2)
    dv_all = jnp.moveaxis(dv_all_rev, 0, 2)
    return dh_all, dh0, dv_all
