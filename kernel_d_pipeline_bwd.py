"""
Milestone 3 -- "Kernel D" (deliberately plain JAX, not Pallas -- see chat for
rationale) + full pipeline glue chaining Kernels A -> B -> C -> D.

Math (chunk_gated_delta_rule_fwd_kernel_h_blockdim64, chunk_kda.py L977-1263,
plus the derived output-combine formula from Milestone 1, both already
validated against the token-serial ground truth):
    v_new   = u - w_pseudo @ h_pre
    o       = scale * (qg @ h_pre) + Aqk @ v_new
    h_new   = h_pre * exp(gc_last) + kg^T @ v_new
sequential over chunks, h_pre carried via jax.lax.scan.

NOT YET TESTED ON REAL TPU (though this part reuses ordinary XLA ops with no
Mosaic/Pallas involvement, so it should be much lower-risk than Kernels A-C).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from kernel_a_scores import build_chunk_scores_pallas, BT
from kernel_b_solve import wy_solve_pallas
from kernel_c_recompute import recompute_wy_pallas

_HIGHEST = jax.lax.Precision.HIGHEST


def gdn2_inter_chunk_combine(Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=None):
    """Aqk: (B,H,n_chunks,BT,BT). w_pseudo,u,kg,qg: (B,H,n_chunks,BT,D).
    gc_last: (B,H,n_chunks,D). Returns o: (B,H,n_chunks,BT,D), h_final: (B,H,D,D),
    h_pre_all: (B,H,n_chunks,D,D) (state ENTERING each chunk, needed by backward
    Milestone B1/B2 -- chunk_gated_delta_rule_bwd_dhu / chunk_kda_bwd_dAv both
    need per-chunk h/v_new, not just the final state), v_new_all: (B,H,n_chunks,BT,D).
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
        # Defensive clip/sanitize -- see incident notes in BACKWARD_PLAN.md
        # (real training, step 710+): GDN-2 has documented history of state
        # blowing up to non-finite; this mirrors the old associative_scan
        # path's Frobenius-clip/nan_to_num, which the WY-chunked state update
        # has no equivalent of on its own.
        h_new = jnp.nan_to_num(jnp.clip(h_new, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
        o_c = jnp.nan_to_num(o_c, nan=0.0, posinf=1e4, neginf=-1e4)

        return h_new, (o_c, h_pre, v_new)

    h_final, (o_scanned, h_pre_scanned, v_new_scanned) = jax.lax.scan(step, h0, to_scan)
    o = jnp.moveaxis(o_scanned, 0, 2)  # (B,H,n_chunks,BT,D)
    h_pre_all = jnp.moveaxis(h_pre_scanned, 0, 2)  # (B,H,n_chunks,D,D) -- state ENTERING each chunk
    v_new_all = jnp.moveaxis(v_new_scanned, 0, 2)  # (B,H,n_chunks,BT,D)
    return o, h_final, h_pre_all, v_new_all


def gdn2_pallas_forward(q, k, v, w, b, g, scale, h0=None):
    """Full staged pipeline: Kernel A (Pallas) -> B (Pallas) -> C (Pallas)
    -> D (plain JAX scan). q,k,v,w,b,g: (B,L,H,D). Returns o: (B,L,H,D),
    h_final: (B,H,D,D).

    NOTE: no shard_map here. An earlier version of this function added one,
    but that's WRONG if the caller (GatedDeltaNet2J in model.py) already
    wraps the whole gdn2_pallas_forward_trainable call in its own shard_map
    (matching MLAJ's pallas_flash_attention pattern) -- nesting two shard_map
    calls on the same mesh axis fails ("context mesh ... should match the
    mesh passed to shard_map", confirmed on real v5e-8: outer shard_map puts
    the axis into Manual mode, inner shard_map then tries to open it again
    with a plain Auto mesh). Sharding is the CALLER's responsibility -- do it
    once, at the GatedDeltaNet2J call site, not inside this function.
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
    A = wy_solve_pallas(Akk)
    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
    o_chunks, h_final, h_pre_all, v_new_all = gdn2_inter_chunk_combine(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0
    )

    n_chunks = L // BT
    o = jnp.moveaxis(o_chunks, 1, 3)          # (B,n_chunks,BT,H,D)
    o = o.reshape(bsz, n_chunks * BT, H, D)     # (B,L,H,D)
    # h_pre_all/v_new_all/Aqk kept around for the future backward Milestones
    # (B1/B2 need per-chunk h_pre and v_new; см. BACKWARD_PLAN.md) -- NOT
    # returned by default to avoid breaking the existing (o, h_final) call
    # sites (model.py, all current tests). Use gdn2_pallas_forward_with_residuals
    # below when backward Pallas kernels need them.
    return o, h_final


def gdn2_pallas_forward_with_residuals(q, k, v, w, b, g, scale, h0=None):
    """Same as gdn2_pallas_forward, but also returns the per-chunk residuals
    (Aqk, h_pre_all, v_new_all, ...) needed by the backward Pallas kernels
    being developed per BACKWARD_PLAN.md (Milestone B1/B2 onward). Kept
    SEPARATE from gdn2_pallas_forward so backward development can't
    accidentally affect the production forward call path (model.py / all
    existing tests use gdn2_pallas_forward, unchanged, zero risk).
    """
    bsz, L, H, D = q.shape
    if h0 is None:
        h0 = jnp.zeros((bsz, H, D, D), dtype=jnp.float32)

    Aqk, Akk = build_chunk_scores_pallas(q, k, b, g, scale)
    A = wy_solve_pallas(Akk)
    w_pseudo, u, kg, qg, gc_last = recompute_wy_pallas(q, k, v, w, b, g, A)
    o_chunks, h_final, h_pre_all, v_new_all = gdn2_inter_chunk_combine(
        Aqk, w_pseudo, u, kg, qg, gc_last, scale, h0=h0
    )
    n_chunks = L // BT
    o = jnp.moveaxis(o_chunks, 1, 3).reshape(bsz, n_chunks * BT, H, D)
    residuals = dict(Aqk=Aqk, Akk=Akk, A=A, h_pre_all=h_pre_all, v_new_all=v_new_all,
                      w_pseudo=w_pseudo, u=u, kg=kg, qg=qg, gc_last=gc_last)
    return o, h_final, residuals
