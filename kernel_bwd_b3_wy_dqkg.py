"""
Milestone B3 (BACKWARD_PLAN.md) -- WY/dqkg fused backward. STAGE 1 of the
project's standard pipeline ("вывод из нашей математики -> numpy/jax.vjp
проверка -> Pallas -> TPU-тест"): this file is the formula + jax.vjp
cross-check ONLY. No Pallas yet -- port to Pallas only after this has been
run and confirmed (I have no JAX/TPU access in this sandbox, see note at
bottom of file).

Reference: chunk_gdn2.py, chunk_gdn2_bwd_kernel_wy_dqkg_fused (1248-1421).

============================================================================
SCOPE (what this milestone owns, matching the plan's Вход/Выход):
============================================================================
Forward pieces being differentiated (from kernel_c_recompute.py /
gdn2_wy_reference.py::_build_chunk_wy + kernel_d_pipeline.py's chunk_step),
for ONE chunk, with A (== WY-solved (I+Akk)^-1) and gc (chunk-local
cumulative decay, already computed -- cumsum itself is B5's job, not this
milestone's) treated as already-known/leaf quantities:

    kb_decayed = b*k*exp(gc)
    w_pseudo   = A @ kb_decayed
    u          = A @ (w*v)
    kg         = k*exp(gc_last - gc)          # gc_last = gc[-1,:], NOT independent
    qg         = q*exp(gc)
    wh         = w_pseudo @ h_pre
    v_new      = u - wh
    qh         = qg @ h_pre
    write      = kg^T @ v_new                  # feeds h_new = h_pre*decay + write

Upstream cotangents this milestone RECEIVES (already produced by B1+B2, do
not recompute them here):
    do      : dL/do -- only the qh-term's share matters here (scale*do),
              the intra=Aqk@v_new term's share was already folded into
              Milestone B2's dv_partial.
    dv      : dL/dv_new, TOTAL (B1's dv_write + B2's dv_partial, already summed).
    dh_next : dL/dh_new for THIS chunk == the "dh_carry" value B1's reverse
              scan used as its INPUT for this chunk == dh_all[c+1] from B1,
              or dht (the external cotangent on h_final) for the last chunk.
              NOTE: this is NOT the same as B1's dh_all[c] output (dh_pre_c,
              which flows to the PREVIOUS chunk) -- easy to mix up, see
              orchestration note at the bottom of this file.

Outputs (RAW/PARTIAL -- Milestone B4 adds more to dq/dk/db/dgc via the
Aqk/Akk score-build backward; B5 turns dgc into dg_raw):
    dq, dk, db, dw (raw w-gate), dv_raw (gradient on the ORIGINAL v input,
    distinct from the `dv` argument which is dL/dv_new), dgc (partial),
    dAkk (via the explicit matrix-inverse-gradient formula, NOT by
    re-inverting/autodiffing through the solve).

============================================================================
DERIVATION (by hand, all plain product/chain rule -- see verify function
below for the jax.vjp cross-check before trusting any of this):
============================================================================
Local cotangents on the three "junction" values:
    dqh_up   = scale * do                                   (qh = qg @ h_pre)
    dqg      = dqh_up @ h_pre^T                                (C,D)

    dwh      = -dv                                          (v_new = u - wh)
    dw_pseudo= dwh @ h_pre^T = -dv @ h_pre^T                    (C,D)
    du       = dv                                             (C,Dv)

    dkg      = v_new @ dh_next^T          (write = kg^T @ v_new)   (C,D)

Through A (used in BOTH w_pseudo = A@kb_decayed and u = A@wv):
    dA_from_w    = dw_pseudo @ kb_decayed^T                     (C,C)
    dkb_decayed  = A^T @ dw_pseudo                                (C,D)
    dA_from_u    = du @ wv^T                                       (C,C)
    dwv          = A^T @ du                                          (C,Dv)
    dA_total     = dA_from_w + dA_from_u

Matrix-inverse-gradient formula for A=(I+Akk)^-1 (BACKWARD_PLAN.md line
~89/162, already read out of chunk_gdn2_bwd_kernel_wy_dqkg_fused lines
1415-1419 -- standard adjoint of matrix inversion, d(M^-1) = -M^-1 dM M^-1):
    dAkk_raw = -A^T @ dA_total @ A^T                                (C,C)
    dAkk     = dAkk_raw * strict_lower_mask   (Akk structurally zero
               elsewhere -- matches Kernel A's own masking convention)

Elementwise gate splits (product rule, kb_decayed = b*k*exp(gc)):
    dk_from_kb  = dkb_decayed * exp(gc) * b
    db          = dkb_decayed * exp(gc) * k
    dgc_from_kb = dkb_decayed * kb_decayed        (since d(exp(gc))/dgc=exp(gc))

kg = k*exp(gc_last - gc), let x = gc_last - gc:
    dx          = dkg * kg                        (d(exp(x))/dx=exp(x), times k)
    dk_from_kg  = dkg * exp(gc_last - gc)
    dgc_from_kg = -dx                             (dx/dgc = -1, elementwise)
    dgc_last    = sum_i dx[i, :]                  (gc_last broadcasts over i;
                  gc_last IS gc[-1,:] literally, not an independent leaf, so
                  this must be ADDED INTO dgc's last row, not returned
                  separately -- see orchestration note)

qg = q*exp(gc):
    dq          = dqg * exp(gc)
    dgc_from_qg = dqg * qg

wv = w*v (elementwise, w is the write-gate, v the original value input):
    dw     = dwv * v            (raw w-gate gradient)
    dv_raw = dwv * w             (gradient on ORIGINAL v -- distinct from
                                  the `dv` argument, which was dL/dv_new)

Total (partial) dk = dk_from_kb + dk_from_kg  (B4 adds the Akk/Aqk-build share)
Total (partial) dgc = dgc_from_kb + dgc_from_qg + dgc_from_kg, with an extra
    += sum_i(dkg*kg)[i,:] added into ROW -1 ONLY (the gc_last contribution).

NOT YET TESTED (no JAX/TPU access in this sandbox -- see bottom of file).
Everything above is derived by hand; do not trust it until
verify_wy_dqkg_backward() has actually been run and passes.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

_HIGHEST = jax.lax.Precision.HIGHEST


def wy_dqkg_backward_formula_full(q_c, k_c, b_c, w_c, v_c, gc, A, Akk, h_pre,
                                   v_new, do, dv, dh_next, scale):
    """Same as above docstring, but explicit about v_c (ORIGINAL v input,
    distinct from v_new) -- this is the real entry point, use this one.
    """
    C = q_c.shape[0]
    gc_last = gc[-1]  # (D,)

    kb_decayed = b_c * k_c * jnp.exp(gc)          # (C,D)
    wv = w_c * v_c                                   # (C,Dv)
    kg = k_c * jnp.exp(gc_last[None, :] - gc)      # (C,D)
    qg = q_c * jnp.exp(gc)                            # (C,D)

    # --- junction cotangents ---
    dqh_up = scale * do
    dqg = jnp.dot(dqh_up, h_pre.T, precision=_HIGHEST)              # (C,D)

    dwh = -dv
    dw_pseudo = jnp.dot(dwh, h_pre.T, precision=_HIGHEST)             # (C,D)
    du = dv                                                              # (C,Dv)

    dkg = jnp.dot(v_new, dh_next.T, precision=_HIGHEST)                  # (C,D)

    # --- through A ---
    dA_from_w = jnp.dot(dw_pseudo, kb_decayed.T, precision=_HIGHEST)      # (C,C)
    dkb_decayed = jnp.dot(A.T, dw_pseudo, precision=_HIGHEST)               # (C,D)

    dA_from_u = jnp.dot(du, wv.T, precision=_HIGHEST)                         # (C,C)
    dwv = jnp.dot(A.T, du, precision=_HIGHEST)                                  # (C,Dv)

    dA_total = dA_from_w + dA_from_u

    # --- matrix-inverse-gradient formula, A = (I+Akk)^-1 ---
    dAkk_raw = -jnp.dot(A.T, jnp.dot(dA_total, A.T, precision=_HIGHEST), precision=_HIGHEST)
    idx = jnp.arange(C)
    strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
    dAkk = dAkk_raw * strict

    # --- elementwise gate splits ---
    dk_from_kb = dkb_decayed * jnp.exp(gc) * b_c
    db = dkb_decayed * jnp.exp(gc) * k_c
    dgc_from_kb = dkb_decayed * kb_decayed

    dx = dkg * kg                       # x = gc_last - gc
    dk_from_kg = dkg * jnp.exp(gc_last[None, :] - gc)
    dgc_from_kg = -dx
    dgc_last_contrib = jnp.sum(dx, axis=0)  # (D,) -- gc_last broadcasts over i

    dq = dqg * jnp.exp(gc)
    dgc_from_qg = dqg * qg

    dw = dwv * v_c
    dv_raw = dwv * w_c

    dk = dk_from_kb + dk_from_kg
    dgc = dgc_from_kb + dgc_from_qg + dgc_from_kg
    dgc = dgc.at[-1].add(dgc_last_contrib)
    # NOTE for the future Pallas port: `.at[].add()` is fine here (plain
    # JAX, not inside a Pallas kernel body). Once this is ported to Pallas,
    # per project lesson #9 (see HANDOFF.md sec 5), this single-row add on a
    # ref must become an explicit read-modify-write
    # (ref[C-1:C,:] = ref[C-1:C,:] + contrib), NOT `.at[].add()` --
    # scatter-add does not lower on Mosaic/TPU, even for this simple a case.

    dq = jnp.nan_to_num(dq, nan=0.0, posinf=1e4, neginf=-1e4)
    dk = jnp.nan_to_num(dk, nan=0.0, posinf=1e4, neginf=-1e4)
    db = jnp.nan_to_num(db, nan=0.0, posinf=1e4, neginf=-1e4)
    dw = jnp.nan_to_num(dw, nan=0.0, posinf=1e4, neginf=-1e4)
    dv_raw = jnp.nan_to_num(dv_raw, nan=0.0, posinf=1e4, neginf=-1e4)
    dgc = jnp.nan_to_num(dgc, nan=0.0, posinf=1e4, neginf=-1e4)
    dAkk = jnp.nan_to_num(dAkk, nan=0.0, posinf=1e4, neginf=-1e4)

    return dict(dq=dq, dk=dk, db=db, dw=dw, dv_raw=dv_raw, dgc=dgc, dAkk=dAkk)


# ==========================================================================
# jax.vjp cross-check. Isolated piece of the graph, cotangents on
# (qh, v_new, write) supplied DIRECTLY (matching how B1/B2 were validated --
# we already trust those upstream cotangents, no need to re-derive them
# here), so this test validates ONLY this milestone's own math.
# ==========================================================================
def _piece1_forward(q_, k_, b_, w_, v_, gc_, Akk_, h_pre_, *, scale):
    """qh + v_new ONLY -- deliberately EXCLUDES write=kg^T@v_new.

    Why split (found via a real jax.vjp mismatch, see chat -- the original
    single-piece version double-counted v_new's gradient): in the REAL
    pipeline, the `dv` this milestone receives from B1+B2 is already the
    TOTAL dL/d(v_new) -- B1 already added the write-path's contribution
    (dv_write = kg @ dh_next) into it. If `write` were included in THIS
    piece too, jax.vjp would independently rediscover that same
    contribution via autodiff through v_new -> write, and ADD it on top of
    the already-total injected cotangent -- double counting. Keeping
    `write` out of this piece means the injected `dv_up` cotangent is
    consumed exactly once, matching how B3 actually receives it.

    gc_ is a LEAF (cumsum is B5's job). gc_last is NOT independent -- it's
    literally gc_[-1], same as real forward.
    """
    C = q_.shape[0]
    A_ = jnp.linalg.inv(jnp.eye(C, dtype=Akk_.dtype) + Akk_)
    gc_last_ = gc_[-1]

    kb_decayed = b_ * k_ * jnp.exp(gc_)
    w_pseudo = jnp.dot(A_, kb_decayed, precision=_HIGHEST)
    u = jnp.dot(A_, w_ * v_, precision=_HIGHEST)
    qg_ = q_ * jnp.exp(gc_)

    wh = jnp.dot(w_pseudo, h_pre_, precision=_HIGHEST)
    v_new_ = u - wh
    qh = jnp.dot(qg_, h_pre_, precision=_HIGHEST)
    return qh, v_new_


def _piece2_forward(k_, gc_, v_new_):
    """write=kg^T@v_new ONLY, with v_new_ treated as a LEAF (not recomputed
    through A) -- isolates the dkg/dk_from_kg/dgc_from_kg/dgc_last formula
    from the piece1 double-counting issue above. This mirrors how B1 itself
    already validated dv_write = kg @ dh_next against jax.vjp on the real
    gdn2_inter_chunk_combine (see kernel_bwd_b1_dhu.py) -- not re-testing
    that path here, only this milestone's OWN dkg/dk/dgc formula.
    """
    gc_last_ = gc_[-1]
    kg_ = k_ * jnp.exp(gc_last_[None, :] - gc_)
    write = jnp.dot(kg_.T, v_new_, precision=_HIGHEST)
    return write


def verify_wy_dqkg_backward(key, C=8, D=6, Dv=5, scale=0.7, akk_scale=0.05):
    """Pure-JAX/numpy sanity check (no Pallas, no TPU needed): the hand-
    derived wy_dqkg_backward_formula_full must match jax.vjp, split into
    two independent isolated pieces (_piece1_forward, _piece2_forward) to
    avoid the double-counting trap documented on _piece1_forward -- each
    piece is checked against jax.vjp separately, then the formula's summed
    outputs (dk = dk_from_kb+dk_from_kg, dgc = ...) are compared against the
    SUM of both pieces' reference gradients for that quantity.

    akk_scale keeps Akk small so (I+Akk) stays well-conditioned for this
    correctness check -- this is about validating the ALGEBRA, not stress-
    testing the WY-solve's numerical range (that's the separate open
    question in HANDOFF.md sec 6 about BC/stability).

    Returns a dict of per-output relative errors (dq, dk, db, dw, dv_raw,
    dgc, dAkk). Run this locally (no JAX in the sandbox that wrote this --
    please run it and report back, same workflow as every other milestone
    in this project).
    """
    keys = jax.random.split(key, 12)
    q_c = jax.random.normal(keys[0], (C, D))
    k_c = jax.random.normal(keys[1], (C, D))
    b_c = jax.random.normal(keys[2], (C, D))
    w_c = jax.random.normal(keys[3], (C, Dv))
    v_c = jax.random.normal(keys[4], (C, Dv))
    gc = jnp.cumsum(jax.random.normal(keys[5], (C, D)) * 0.1, axis=0)
    h_pre = jax.random.normal(keys[6], (D, Dv))

    idx = jnp.arange(C)
    strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
    Akk = jax.random.normal(keys[7], (C, C)) * akk_scale * strict

    do = jax.random.normal(keys[8], (C, Dv))
    dv_up = jax.random.normal(keys[9], (C, Dv))       # injected as TOTAL dL/d(v_new)
    dh_next = jax.random.normal(keys[10], (D, Dv))    # injected as dL/d(write)

    # --- piece 1: qh, v_new (through A) ---
    fwd1 = lambda *a: _piece1_forward(*a, scale=scale)
    (qh_val, v_new_val), vjp_fn1 = jax.vjp(
        fwd1, q_c, k_c, b_c, w_c, v_c, gc, Akk, h_pre
    )
    dq_ref, dk1_ref, db_ref, dw_ref, dv_raw_ref, dgc1_ref, dAkk_ref_raw, dh_pre_ref1 = vjp_fn1(
        (scale * do, dv_up)
    )
    # Same fix already needed for B2's dAqk_ref (see BACKWARD_PLAN.md progress
    # notes): Akk is structurally strict-lower-triangular, but jnp.linalg.inv
    # differentiates the FULL (C,C) matrix with no notion of that constraint,
    # so dAkk_ref has nonzero entries on/above the diagonal too. Our formula
    # masks dAkk to strict-lower (matching Kernel A/B's own masking), so the
    # reference needs the same mask before comparing -- otherwise the diff is
    # dominated by entries that were never meant to carry gradient.
    dAkk_ref = dAkk_ref_raw * strict

    # --- piece 2: write, v_new as a LEAF (not through A) ---
    (write_val,), vjp_fn2 = jax.vjp(
        lambda k_, gc_, v_new_: (_piece2_forward(k_, gc_, v_new_),), k_c, gc, v_new_val
    )
    dk2_ref, dgc2_ref, dv_new_from_write_ref = vjp_fn2((dh_next,))

    dk_ref = dk1_ref + dk2_ref
    dgc_ref = dgc1_ref + dgc2_ref

    A_val = jnp.linalg.inv(jnp.eye(C) + Akk)
    out = wy_dqkg_backward_formula_full(
        q_c, k_c, b_c, w_c, v_c, gc, A_val, Akk, h_pre,
        v_new_val, do, dv_up, dh_next, scale,
    )

    def relerr(a, b):
        return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(b)) + 1e-12))

    errs = dict(
        dq=relerr(out["dq"], dq_ref),
        dk=relerr(out["dk"], dk_ref),
        db=relerr(out["db"], db_ref),
        dw=relerr(out["dw"], dw_ref),
        dv_raw=relerr(out["dv_raw"], dv_raw_ref),
        dgc=relerr(out["dgc"], dgc_ref),
        dAkk=relerr(out["dAkk"], dAkk_ref),
    )
    return errs


if __name__ == "__main__":
    all_ok = True
    for i, cfg in enumerate([
        dict(C=8, D=6, Dv=5, scale=0.7, akk_scale=0.05),
        dict(C=16, D=8, Dv=8, scale=1.0, akk_scale=0.02),
        dict(C=32, D=16, Dv=16, scale=0.5, akk_scale=0.1),
        dict(C=8, D=4, Dv=6, scale=0.3, akk_scale=0.01),
    ]):
        errs = verify_wy_dqkg_backward(jax.random.PRNGKey(100 + i), **cfg)
        print(f"cfg{i} {cfg} -> {errs}")
        if not all(v < 1e-4 for v in errs.values()):
            all_ok = False
    if all_ok:
        print("B3 formula matches jax.vjp on the isolated piece across all configs. "
              "Next: run this on your machine (no JAX here), then port to Pallas "
              "kernel_bwd_b3_wy_dqkg.py's _kernel_b3_body once confirmed.")
    else:
        print("MISMATCH -- do not port to Pallas yet, formula needs fixing first.")


# ==========================================================================
# STAGE 3: Pallas/TPU port. Only reached because verify_wy_dqkg_backward()
# above was run on real hardware and passed (rel err ~1e-7/1e-8 across all
# 4 configs, incl. the dAkk-masking fix -- see chat). NOT YET TESTED ON
# REAL TPU ITSELF (this port, unlike the formula above, has never been
# executed anywhere -- please run test_kernel_bwd_b3.py-equivalent on your
# v5e-8 and report back, same workflow as every other Pallas kernel here).
#
# Unlike Kernel A/B4 (kernel_a_scores.py, kernel_bwd_b4_intra.py), this
# kernel does NOT need the unrolled si,sj sub-block loop over BC=128 pairs:
# every matmul in this formula is a plain 2-operand (BT,BT)@(BT,D) or
# (BT,D)@(D,BT) contraction (A@kb_decayed-style, same shape pattern already
# proven to lower fine in Kernel B/C) -- there is no 3-way (BC,BC,D)
# decay-difference product here (that only exists in the Aqk/Akk SCORE-BUILD
# formula, i.e. Kernel A / Milestone B4, which this milestone does not
# recompute -- A and Akk arrive here as already-known values). So the whole
# per-chunk computation runs as one straight-line block, same style as
# kernel_bwd_b2_dav.py.
#
# In this codebase d_head is used for BOTH the key axis and the value axis
# (single head dim -- see gdn2_wy_reference.py's own docstring), so D==Dv
# throughout; only one (BT,D) BlockSpec is needed for q/k/b/w/v/do/dv/etc.
#
# The one real Mosaic trap here (lesson #9, kernel_bwd_b4_intra.py): the
# formula's dgc = ... ; dgc[-1] += sum_i(dx) step. `.at[-1].add(...)` on a
# plain JAX value is a scatter-add -- doesn't lower on Mosaic even with a
# fully static index. Fixed the same way B4 fixed it: write dgc to the
# output ref first, then do an explicit read-modify-write on JUST the last
# row of the ref (load, add, store -- two ordinary ops, not a scatter
# primitive).
# ==========================================================================
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from kernel_a_scores import BT


def _kernel_b3_body(q_ref, k_ref, b_ref, w_ref, v_ref, gc_ref, a_ref, akk_ref,
                     hpre_ref, vnew_ref, do_ref, dv_ref, dhnext_ref,
                     dq_ref, dk_ref, db_ref, dw_ref, dvraw_ref, dgc_ref, dakk_ref,
                     *, scale):
    q_c = q_ref[0, 0, 0].astype(jnp.float32)      # (BT, D)
    k_c = k_ref[0, 0, 0].astype(jnp.float32)
    b_c = b_ref[0, 0, 0].astype(jnp.float32)
    w_c = w_ref[0, 0, 0].astype(jnp.float32)      # (BT, D) (D==Dv, see module note)
    v_c = v_ref[0, 0, 0].astype(jnp.float32)
    gc = gc_ref[0, 0, 0].astype(jnp.float32)      # (BT, D)
    A = a_ref[0, 0, 0].astype(jnp.float32)        # (BT, BT)
    Akk = akk_ref[0, 0, 0].astype(jnp.float32)    # (BT, BT), used only for its shape/mask
    h_pre = hpre_ref[0, 0, 0].astype(jnp.float32)  # (D, D)
    v_new = vnew_ref[0, 0, 0].astype(jnp.float32)  # (BT, D)
    do = do_ref[0, 0, 0].astype(jnp.float32)        # (BT, D)
    dv = dv_ref[0, 0, 0].astype(jnp.float32)          # (BT, D) -- TOTAL dL/d(v_new), from B1+B2
    dh_next = dhnext_ref[0, 0, 0].astype(jnp.float32)  # (D, D)

    C = BT
    gc_last = gc[C - 1]  # (D,) -- static index, gc_last is literally gc's last row

    kb_decayed = b_c * k_c * jnp.exp(gc)
    kg = k_c * jnp.exp(gc_last[None, :] - gc)
    qg = q_c * jnp.exp(gc)
    wv = w_c * v_c

    # --- junction cotangents ---
    dqh_up = scale * do
    dqg = jnp.dot(dqh_up, h_pre.T, precision=_HIGHEST)               # (BT, D)

    dwh = -dv
    dw_pseudo = jnp.dot(dwh, h_pre.T, precision=_HIGHEST)              # (BT, D)
    du = dv                                                              # (BT, D)

    dkg = jnp.dot(v_new, dh_next.T, precision=_HIGHEST)                   # (BT, D)

    # --- through A ---
    dA_from_w = jnp.dot(dw_pseudo, kb_decayed.T, precision=_HIGHEST)       # (BT, BT)
    dkb_decayed = jnp.dot(A.T, dw_pseudo, precision=_HIGHEST)                # (BT, D)

    dA_from_u = jnp.dot(du, wv.T, precision=_HIGHEST)                          # (BT, BT)
    dwv = jnp.dot(A.T, du, precision=_HIGHEST)                                   # (BT, D)

    dA_total = dA_from_w + dA_from_u

    # --- matrix-inverse-gradient formula, A = (I+Akk)^-1 ---
    dAkk_raw = -jnp.dot(A.T, jnp.dot(dA_total, A.T, precision=_HIGHEST), precision=_HIGHEST)
    idx = jnp.arange(C)
    strict = (idx[:, None] > idx[None, :]).astype(jnp.float32)
    dAkk = dAkk_raw * strict

    # --- elementwise gate splits ---
    dk_from_kb = dkb_decayed * jnp.exp(gc) * b_c
    db = dkb_decayed * jnp.exp(gc) * k_c
    dgc_from_kb = dkb_decayed * kb_decayed

    dx = dkg * kg
    dk_from_kg = dkg * jnp.exp(gc_last[None, :] - gc)
    dgc_from_kg = -dx
    dgc_last_contrib = jnp.sum(dx, axis=0)  # (D,)

    dq = dqg * jnp.exp(gc)
    dgc_from_qg = dqg * qg

    dw = dwv * v_c
    dv_raw = dwv * w_c

    dk = dk_from_kb + dk_from_kg
    dgc = dgc_from_kb + dgc_from_qg + dgc_from_kg  # last-row gc_last contribution NOT yet added

    dq_ref[0, 0, 0] = jnp.nan_to_num(dq, nan=0.0, posinf=1e4, neginf=-1e4)
    dk_ref[0, 0, 0] = jnp.nan_to_num(dk, nan=0.0, posinf=1e4, neginf=-1e4)
    db_ref[0, 0, 0] = jnp.nan_to_num(db, nan=0.0, posinf=1e4, neginf=-1e4)
    dw_ref[0, 0, 0] = jnp.nan_to_num(dw, nan=0.0, posinf=1e4, neginf=-1e4)
    dvraw_ref[0, 0, 0] = jnp.nan_to_num(dv_raw, nan=0.0, posinf=1e4, neginf=-1e4)
    dakk_ref[0, 0, 0] = jnp.nan_to_num(dAkk, nan=0.0, posinf=1e4, neginf=-1e4)

    dgc_ref[0, 0, 0] = jnp.nan_to_num(dgc, nan=0.0, posinf=1e4, neginf=-1e4)
    # ФИКС (см. lesson #9 / kernel_bwd_b4_intra.py): gc_last = gc[-1] is not
    # an independent leaf, its contribution must be ADDED into dgc's last
    # row -- `.at[-1].add()` on a plain value is a scatter-add and does not
    # lower on Mosaic even with a static index. Explicit ref read-modify-
    # write instead (load the row just written above, add, store back) --
    # ordinary load+store, not a scatter primitive.
    last_row = dgc_ref[0, 0, 0, C - 1:C, :]
    dgc_ref[0, 0, 0, C - 1:C, :] = jnp.nan_to_num(
        last_row + dgc_last_contrib[None, :], nan=0.0, posinf=1e4, neginf=-1e4
    )


def wy_dqkg_backward_pallas(q, k, b, w, v, gc, A, Akk, h_pre_all, v_new_all,
                             do, dv, dh_next_all, scale):
    """q,k,b,w,v,gc,do,dv: (B,H,n_chunks,BT,D) (D==Dv in this codebase).
    A,Akk: (B,H,n_chunks,BT,BT). h_pre_all, dh_next_all: (B,H,n_chunks,D,D)
    -- h_pre_all is the forward-saved per-chunk incoming state (from the B1
    precondition patch, kernel_d_pipeline_PATCH.py); dh_next_all is the
    PER-CHUNK dh_next described in this file's module docstring (== B1's
    dh_all shifted by one chunk, with dht at the last chunk -- caller's
    responsibility to build this, see orchestration note below).

    Returns dict: dq, dk, db, dw, dv_raw, dgc (B,H,n_chunks,BT,D), dAkk
    (B,H,n_chunks,BT,BT). All still PARTIAL -- Milestone B4 adds the
    Aqk/Akk-build contributions to dq/dk/db/dgc; B5 turns dgc into dg_raw.
    """
    bsz, H, n_chunks, _BT, D = q.shape
    grid = (bsz, H, n_chunks)

    io_spec = pl.BlockSpec((1, 1, 1, BT, D), lambda i, h, c: (i, h, c, 0, 0))
    score_spec = pl.BlockSpec((1, 1, 1, BT, BT), lambda i, h, c: (i, h, c, 0, 0))
    h_spec = pl.BlockSpec((1, 1, 1, D, D), lambda i, h, c: (i, h, c, 0, 0))

    dq, dk, db, dw, dv_raw, dgc, dAkk = pl.pallas_call(
        lambda *refs: _kernel_b3_body(*refs, scale=scale),
        grid=grid,
        in_specs=[io_spec, io_spec, io_spec, io_spec, io_spec, io_spec,
                   score_spec, score_spec, h_spec, io_spec, io_spec, io_spec, h_spec],
        out_specs=[io_spec, io_spec, io_spec, io_spec, io_spec, io_spec, score_spec],
        out_shape=[
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, D), jnp.float32),
            jax.ShapeDtypeStruct((bsz, H, n_chunks, BT, BT), jnp.float32),
        ],
        # Kernel B4 needed 150MiB for its live (BC,BC,D) intermediates; this
        # kernel has no such 3-way tensors (see module note -- everything is
        # a plain 2-operand matmul), but does carry several simultaneously-
        # live (BT,BT)=256x256 fp32 intermediates (A, Akk, dA_from_w,
        # dA_from_u, dA_total, dAkk_raw, dAkk, strict) plus a handful of
        # (BT,D) ones. 100MiB (same as Kernel A) as a starting budget --
        # tune down after a real profiling run if excessive.
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=100 * 1024 * 1024),
    )(q, k, b, w, v, gc, A, Akk, h_pre_all, v_new_all, do, dv, dh_next_all)

    return dict(dq=dq, dk=dk, db=db, dw=dw, dv_raw=dv_raw, dgc=dgc, dAkk=dAkk)
