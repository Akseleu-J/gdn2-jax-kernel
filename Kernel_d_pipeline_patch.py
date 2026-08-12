"""
PATCH for kernel_d_pipeline.py -- Backward precondition (BACKWARD_PLAN.md,
"Предварительное условие", option (a), recommended).

Add this function to kernel_d_pipeline.py ALONGSIDE the existing
gdn2_inter_chunk_combine (do not delete/modify that one -- forward already
validated on real TPU, leave it untouched). This is a pure additive-output
variant: identical forward math, scan step additionally threads out
(h_pre, v_new) per chunk. Cannot change forward numerics -- same computation,
just also returned.

gdn2_pallas_forward_trainable (kernel_trainable.py) should NOT switch to this
variant for its own forward call (keeps the existing, already-proven Pallas
pipeline as-is for inference/training forward). This variant is ONLY needed
as the backward path's forward-residual-recompute inside a future fused
custom_vjp (Milestone B6) -- i.e. called from the *backward* function, on
the pure-JAX WY reference side, exactly where gdn2_chunked_wy_reference is
already re-traced today for jax.vjp. Kept separate/parallel to
gdn2_inter_chunk_combine so nothing about the currently-training forward
path changes.

Validated (CPU/plain JAX, no TPU needed since this is unchanged Kernel-D-style
plain JAX): cross-checked against jax.vjp on an isolated version of this exact
recurrence -- see chat / test_kernel_bwd_b1.py. All formula pieces (dh0, dqg,
dkg, dv_new-write-contribution) matched to float32 roundoff (~1e-7) across
multiple shape configs.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST


def gdn2_inter_chunk_combine_with_state(Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=None):
    """Same math as gdn2_inter_chunk_combine, plus per-chunk h_pre/v_new
    outputs needed by Milestone B1's backward (dhu reverse-scan).

    Aqk: (B,H,n_chunks,BT,BT). w_pseudo,u,kg,qg: (B,H,n_chunks,BT,D).
    gc_last: (B,H,n_chunks,D). Returns:
      o: (B,H,n_chunks,BT,D)
      h_final: (B,H,D,D)
      h_pre_all: (n_chunks,B,H,D,D)  -- state ENTERING each chunk (chunk axis FIRST,
                  matches jax.lax.scan's natural stacking; caller can moveaxis if needed)
      v_new_all: (n_chunks,B,H,BT,D)
    """
    bsz, H, n_chunks, _BT, D = w_pseudo.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    to_scan = tuple(jnp.moveaxis(x, 2, 0) for x in (Aqk, w_pseudo, u, kg, qg, gc_last))

    def step(h_pre, inputs):
        Aqk_c, w_pseudo_c, u_c, kg_c, qg_c, gclast_c = inputs
        wh = jnp.einsum("bhid,bhdv->bhiv", w_pseudo_c, h_pre, precision=_HIGHEST)
        v_new = u_c - wh                                                              # (B,H,BT,D)
        qh = jnp.einsum("bhid,bhdv->bhiv", qg_c, h_pre, precision=_HIGHEST)
        intra = jnp.einsum("bhij,bhjv->bhiv", Aqk_c, v_new, precision=_HIGHEST)
        o_c = scale * qh + intra                                                      # (B,H,BT,D)

        decay_h = jnp.exp(gclast_c)[..., None]                                          # (B,H,D,1)
        write = jnp.einsum("bhid,bhiv->bhdv", kg_c, v_new, precision=_HIGHEST)
        h_new = h_pre * decay_h + write

        # ФИКС (см. HANDOFF.md §6 / BACKWARD_PLAN.md incident, step 710+):
        # same sanitization already applied to h_new/o_c in the main
        # gdn2_inter_chunk_combine -- mirror it here so this variant is no
        # less safe than the one already in production.
        h_new = jnp.nan_to_num(jnp.clip(h_new, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
        o_c = jnp.nan_to_num(o_c, nan=0.0, posinf=1e4, neginf=-1e4)

        return h_new, (o_c, h_pre, v_new)

    h_final, (o_scanned, h_pre_all, v_new_all) = jax.lax.scan(step, h0, to_scan)
    o = jnp.moveaxis(o_scanned, 0, 2)  # (B,H,n_chunks,BT,D)
    return o, h_final, h_pre_all, v_new_all
