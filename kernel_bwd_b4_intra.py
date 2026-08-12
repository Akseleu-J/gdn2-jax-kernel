"""
Milestone B4 (BACKWARD_PLAN.md) -- Pallas TPU kernel for the intra-score-build
backward: given dAqk, dAkk (from B2+B3) and the original chunk inputs q, k, b,
g_raw, produces the score-build's contribution to dq, dk, db, dgc (dgc still
needs Milestone B5's reverse-cumsum to become dg_raw).

Formula: derived by hand and validated in two stages (project convention --
see kernel_bwd_b4_intra.py for the whole-chunk plain-JAX derivation/jax.vjp
check, and subblock_check_fixed.py for the BC-sub-block decomposition check
against that whole-chunk formula, both already run: rel errors ~1e-7-1e-8,
including at the real project size BT=256/BC=128/D=128).

Mirrors Kernel A's (kernel_a_scores.py) structure closely, since this is
literally Kernel A's backward:
  - same si,sj unrolled Python loop over the BT=2*BC lower-triangle sub-block
    pairs (0,0),(1,0),(1,1) -- N_SUB=2 hardcoded, same as Kernel A/B.
  - same tril-ones-matmul trick for cumsum (gc), since cumsum isn't lowered
    by Mosaic.
  - no einsum (Mosaic dot_general dimension-number parser failure, see Kernel
    A's docstring) -- explicit broadcast-multiply + jnp.sum instead.
  - no dynamic_slice/dynamic_update_slice inside loops -- all si/sj/i0/i1/j0/j1
    are static Python ints from the unrolled loop, so ordinary slicing and
    `.at[].add()` on JAX values (not refs) lower fine, same category as
    Kernel A's own static i0:i1 ref writes.
  - explicit vmem_limit_bytes (default 16MB is insufficient for the (BC,BC,D)
    intermediates, same reasoning as Kernel A -- this kernel has MORE live
    (BC,BC,D) intermediates per sub-block than Kernel A's forward, so budget
    is set higher: 150MiB).
  - nan_to_num before writing outputs (project-wide backward sanitization
    convention, see HANDOFF.md sec 6 / BACKWARD_PLAN.md incident notes).

NOT YET TESTED ON REAL TPU.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from kernel_a_scores import BT, BC, N_SUB

_HIGHEST = jax.lax.Precision.HIGHEST
_CLIP = 20.0

assert N_SUB == 2, "Kernel B4 currently implements only the 2-subblock (BT=2*BC) case, matching Kernel A/B."


def _dL_pair_sum(dM, edecay, R):
    """dL[i,d] = sum_j dM[i,j]*edecay[i,j,d]*R[j,d] -> (BC,D)."""
    tmp = dM[:, :, None] * edecay
    tmp = tmp * R[None, :, :]
    return jnp.sum(tmp, axis=1)


def _dR_pair_sum(dM, edecay, L):
    """dR[j,d] = sum_i dM[i,j]*L[i,d]*edecay[i,j,d] -> (BC,D)."""
    tmp = dM[:, :, None] * edecay
    tmp = tmp * L[:, None, :]
    return jnp.sum(tmp, axis=0)


def _dgc_pair_sum(dM, edecay, L, R, clipmask):
    """weight[i,j,d] = dM[i,j]*L[i,d]*R[j,d]*edecay[i,j,d]*clipmask[i,j,d].
    dgc_i[i,d] = sum_j weight;  dgc_j[j,d] = -sum_i weight.
    """
    weight = dM[:, :, None] * L[:, None, :] * R[None, :, :] * edecay * clipmask
    dgc_i = jnp.sum(weight, axis=1)
    dgc_j = -jnp.sum(weight, axis=0)
    return dgc_i, dgc_j


def _kernel_b4_body(q_ref, k_ref, b_ref, g_ref, daqk_ref, dakk_ref,
                     dq_ref, dk_ref, db_ref, dgc_ref, *, scale):
    q_full = q_ref[0, 0, 0].astype(jnp.float32)   # (BT, D)
    k_full = k_ref[0, 0, 0].astype(jnp.float32)
    b_full = b_ref[0, 0, 0].astype(jnp.float32)
    g_raw = g_ref[0, 0, 0].astype(jnp.float32)
    dAqk = daqk_ref[0, 0, 0].astype(jnp.float32)   # (BT, BT)
    dAkk = dakk_ref[0, 0, 0].astype(jnp.float32)

    bt_idx = jnp.arange(BT)
    tril_ones_bt = (bt_idx[:, None] >= bt_idx[None, :]).astype(jnp.float32)
    gc = jnp.dot(tril_ones_bt, g_raw, precision=_HIGHEST)  # same cumsum-via-matmul trick as Kernel A/C

    bk_full = b_full * k_full

    # ФИКС (после реального прогона на TPU): `.at[].add()` on plain JAX
    # values -- even with fully STATIC slice bounds -- lowers to the
    # `scatter-add` primitive, which Mosaic/Pallas-TPU does not implement
    # ("Unimplemented primitive in Pallas TPU lowering for tc: scatter-add").
    # Kernel A/B/C never hit this because they only ever WRITE ONCE to each
    # disjoint sub-block region of an output ref -- no accumulation. Here we
    # genuinely need to accumulate across MULTIPLE (si,sj) pairs into the
    # same (BC,D) region (e.g. dq's si=1 sub-block gets contributions from
    # BOTH (1,0) and (1,1)). Fix: accumulate via explicit ref
    # read-modify-write (load, add, store -- all with static slices), which
    # is just two independent load/store ops, NOT a scatter primitive --
    # exactly the same ref-slice pattern already proven to work in Kernel
    # A/B/C, just done twice (read then write) instead of once (write only).
    #
    # db's "dbk" (b*k-side accumulator, needs a final *k multiply after the
    # loop) reuses db_ref itself as scratch during the loop -- avoids adding
    # a 5th output tensor. Overwritten with the real db value only after
    # dbk's final accumulated value has been read out.
    dq_ref[0, 0, 0] = jnp.zeros_like(q_full)
    dk_ref[0, 0, 0] = jnp.zeros_like(k_full)
    db_ref[0, 0, 0] = jnp.zeros_like(k_full)   # scratch: accumulates dbk here during the loop
    dgc_ref[0, 0, 0] = jnp.zeros_like(g_raw)

    for si in range(N_SUB):
        for sj in range(si + 1):
            i0, i1 = si * BC, (si + 1) * BC
            j0, j1 = sj * BC, (sj + 1) * BC

            q_i = q_full[i0:i1]
            k_j = k_full[j0:j1]
            bk_i = bk_full[i0:i1]
            gc_i = gc[i0:i1]
            gc_j = gc[j0:j1]

            decay_diff = gc_i[:, None, :] - gc_j[None, :, :]  # (BC, BC, D)
            clipmask = ((decay_diff >= -_CLIP) & (decay_diff <= _CLIP)).astype(jnp.float32)
            edecay = jnp.exp(jnp.clip(decay_diff, -_CLIP, _CLIP))

            dM_qk = dAqk[i0:i1, j0:j1]
            dM_kk = dAkk[i0:i1, j0:j1]
            if si == sj:
                idx = jnp.arange(BC)
                causal = (idx[:, None] >= idx[None, :]).astype(jnp.float32)
                strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
                dM_qk = dM_qk * causal
                dM_kk = dM_kk * strict
            # off-diagonal (si>sj): every pair already satisfies i>j globally
            # (both causal and strict automatically true) -- no extra mask,
            # same reasoning as Kernel A's forward.

            L_qk = scale * q_i
            R_qk = k_j
            dL_qk = _dL_pair_sum(dM_qk, edecay, R_qk)
            dR_qk = _dR_pair_sum(dM_qk, edecay, L_qk)
            dgc_i_qk, dgc_j_qk = _dgc_pair_sum(dM_qk, edecay, L_qk, R_qk, clipmask)

            L_kk = bk_i
            R_kk = k_j
            dL_kk = _dL_pair_sum(dM_kk, edecay, R_kk)
            dR_kk = _dR_pair_sum(dM_kk, edecay, L_kk)
            dgc_i_kk, dgc_j_kk = _dgc_pair_sum(dM_kk, edecay, L_kk, R_kk, clipmask)

            dq_ref[0, 0, 0, i0:i1] = dq_ref[0, 0, 0, i0:i1] + dL_qk * scale
            db_ref[0, 0, 0, i0:i1] = db_ref[0, 0, 0, i0:i1] + dL_kk  # scratch: dbk accumulator
            dk_ref[0, 0, 0, j0:j1] = dk_ref[0, 0, 0, j0:j1] + dR_qk + dR_kk
            dgc_ref[0, 0, 0, i0:i1] = dgc_ref[0, 0, 0, i0:i1] + dgc_i_qk + dgc_i_kk
            dgc_ref[0, 0, 0, j0:j1] = dgc_ref[0, 0, 0, j0:j1] + dgc_j_qk + dgc_j_kk

    dbk_final = db_ref[0, 0, 0]           # fully-accumulated dbk, read back out of the scratch
    dk_final = dk_ref[0, 0, 0] + dbk_final * b_full
    db_final = dbk_final * k_full
    dq_final = dq_ref[0, 0, 0]
    dgc_final = dgc_ref[0, 0, 0]

    # ФИКС (см. HANDOFF.md §6 / BACKWARD_PLAN.md incident): same backward-side
    # sanitization convention used throughout this project's backward pieces
    # (kernel_bwd_b1_dhu.py, kernel_b_solve.py etc.) -- non-finite upstream
    # dAqk/dAkk (from B2/B3) or an exploded edecay term should not silently
    # propagate NaN into dq/dk/db/dgc.
    dq_ref[0, 0, 0] = jnp.nan_to_num(jnp.clip(dq_final, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
    dk_ref[0, 0, 0] = jnp.nan_to_num(jnp.clip(dk_final, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
    db_ref[0, 0, 0] = jnp.nan_to_num(jnp.clip(db_final, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)
    dgc_ref[0, 0, 0] = jnp.nan_to_num(jnp.clip(dgc_final, -1e4, 1e4), nan=0.0, posinf=1e4, neginf=-1e4)


def intra_backward_pallas(dAqk, dAkk, q, k, b, g, scale):
    """dAqk, dAkk: (B,H,n_chunks,BT,BT). q,k,b,g: (B,L,H,D) (chunked internally,
    L must be divisible by BT, same convention as build_chunk_scores_pallas).
    Returns dq, dk, db, dgc: (B,H,n_chunks,BT,D) -- dgc still needs
    Milestone B5's reverse-cumsum to become dg_raw.
    """
    bsz, L, H, D = q.shape
    assert D == 128, f"Kernel B4 assumes d_head=128 (MXU tile); got D={D}."
    assert L % BT == 0, f"seq_len={L} must be divisible by BT={BT}."
    n_chunks = L // BT

    def reshape_in(t):
        t = t.reshape(bsz, n_chunks, BT, H, D)
        return jnp.moveaxis(t, (1, 3), (2, 1))  # -> (B, H, n_chunks, BT, D)

    q_r, k_r, b_r, g_r = map(reshape_in, (q, k, b, g))

    grid = (bsz, H, n_chunks)
    io_spec = pl.BlockSpec((1, 1, 1, BT, D), lambda i, h, c: (i, h, c, 0, 0))
    score_spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))

    dq, dk, db, dgc = pl.pallas_call(
        lambda *refs: _kernel_b4_body(*refs, scale=scale),
        grid=grid,
        in_specs=[io_spec, io_spec, io_spec, io_spec, score_spec, score_spec],
        out_specs=[io_spec, io_spec, io_spec, io_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
        ],
        # Kernel A needed 100MiB for its (BC,BC,D) forward intermediates.
        # This backward kernel has MORE simultaneously-live (BC,BC,D)-shaped
        # arrays per sub-block iteration (edecay, clipmask, plus the dL/dR/
        # dgc weight tensors for BOTH Aqk and Akk) -- budget raised to
        # 150MiB accordingly. Re-tune down after a real profiling run if
        # this turns out to be excessive.
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=150 * 1024 * 1024),
    )(q_r, k_r, b_r, g_r, dAqk, dAkk)

    return dq, dk, db, dgc
