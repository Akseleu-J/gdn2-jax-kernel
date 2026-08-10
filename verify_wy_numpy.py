"""
Numpy verification of the WY-chunked GDN-2 forward against the token-serial
ground truth from fused_recurrent_gdn2.py.

This is a correctness check of the ALGEBRA (Milestone 1), run in numpy because
the sandbox has no network access to install JAX. The formulas here are a
direct transliteration of:
  - fused_recurrent_gdn2_fwd_kernel (per-token loop, lines 193-244)         -> token_serial()
  - chunk_gdn2 Kernel 1/2/3 (Aqk/Akk build + WY solve)                      -> build_chunk_wy()
  - recompute_w_u_fwd_gdn2_kernel (w, u, kg, qg)                           -> build_chunk_wy()
  - chunk_gated_delta_rule_fwd_kernel_h_blockdim64 (inter-chunk state)      -> chunked_wy()
  - derived output combine o = scale*qg@h_pre + Aqk@v_new                  -> chunked_wy()

Shapes below: single (batch, head) slice, arrays are (T, D) / (T, Dv) for
clarity. Batch/head vmapping is trivial (einsum with leading dims) and left
for the JAX port.
"""
import numpy as np

rng = np.random.default_rng(0)


def token_serial(q, k, v, g, b, w, scale, h0=None):
    """Ground truth: fused_recurrent_gdn2_fwd_kernel, unrolled token-by-token.

    q,k,b: (T, D)   v,w: (T, Dv)   g: (T, D) raw per-token log-decay (natural log)
    Returns o: (T, Dv), h_final: (D, Dv)
    """
    T, D = q.shape
    Dv = v.shape[-1]
    h = np.zeros((D, Dv), dtype=np.float64) if h0 is None else h0.copy()
    o = np.zeros((T, Dv), dtype=np.float64)
    alpha = np.exp(g)  # per-token decay factor, channel-wise on D
    for t in range(T):
        h = h * alpha[t][:, None]                      # S <- Diag(alpha_t) S
        bk = b[t] * k[t]                                 # (D,)
        erase = bk @ h                                    # (Dv,)  = (b*k)^T S
        v_new = w[t] * v[t] - erase                        # (Dv,)
        h = h + np.outer(k[t], v_new)                       # S <- S + k v_new^T
        o[t] = h.T @ (q[t] * scale)                          # o = S^T q
    return o, h


def build_chunk_wy(q, k, v, g_raw, b, w, scale):
    """One chunk: Kernels 1+2+3 (Aqk, Akk, WY solve) + Kernel 4 (w_pseudo,u,kg,qg).

    q,k,b: (C, D)   v,w: (C, Dv)   g_raw: (C, D) raw per-token log-decay.
    Returns Aqk (C,C), w_pseudo (C,D), u (C,Dv), kg (C,D), qg (C,D), gc_last (D,)
    """
    C, D = q.shape
    gc = np.cumsum(g_raw, axis=0)  # chunk-local cumulative log-decay, inclusive (chunk_local_cumsum)

    # Kernel 1/2: Aqk[i,j] = scale*sum_d q_i k_j exp(gc_i-gc_j), j<=i
    #             Akk[i,j] = sum_d (b_i k_i) k_j exp(gc_i-gc_j), j<i (strict)
    decay_diff = gc[:, None, :] - gc[None, :, :]          # (C,C,D): gc_i - gc_j
    edecay = np.exp(decay_diff)                             # (C,C,D)
    causal = np.tril(np.ones((C, C)), k=0)                   # j<=i inclusive
    strict = np.tril(np.ones((C, C)), k=-1)                  # j<i strict

    Aqk = scale * np.einsum('id,ijd,jd->ij', q, edecay, k) * causal
    bk = b * k
    Akk = np.einsum('id,ijd,jd->ij', bk, edecay, k) * strict

    # Kernel 3: A = (I + Akk)^{-1}  (Akk strictly lower triangular -> always invertible)
    A = np.linalg.inv(np.eye(C) + Akk)

    # Kernel 4: w_pseudo = A @ (b*k*exp(gc)),  u = A @ (w*v)
    kb_decayed = b * k * np.exp(gc)          # (C,D)
    w_pseudo = A @ kb_decayed                  # (C,D)
    u = A @ (w * v)                              # (C,Dv)

    gc_last = gc[-1]                               # (D,) decay at last position of chunk
    kg = k * np.exp(gc_last[None, :] - gc)           # (C,D) reverse-tail-decayed key
    qg = q * np.exp(gc)                                # (C,D) tail-decayed query

    return Aqk, w_pseudo, u, kg, qg, gc_last


def chunked_wy(q, k, v, g_raw, b, w, scale, chunk_size, h0=None):
    """Full chunked WY forward: chunk loop + inter-chunk state recurrence
    (chunk_gated_delta_rule_fwd_kernel_h_blockdim64) + derived output combine.
    """
    T, D = q.shape
    Dv = v.shape[-1]
    assert T % chunk_size == 0, "T must be divisible by chunk_size for this reference"
    n_chunks = T // chunk_size

    h_pre = np.zeros((D, Dv), dtype=np.float64) if h0 is None else h0.copy()
    o = np.zeros((T, Dv), dtype=np.float64)

    for c in range(n_chunks):
        sl = slice(c * chunk_size, (c + 1) * chunk_size)
        Aqk, w_pseudo, u, kg, qg, gc_last = build_chunk_wy(
            q[sl], k[sl], v[sl], g_raw[sl], b[sl], w[sl], scale
        )
        v_new = u - w_pseudo @ h_pre                 # WY correction using PRE-chunk state
        o[sl] = scale * (qg @ h_pre) + Aqk @ v_new     # inter-chunk + intra-chunk combine

        # inter-chunk state update (chunk_gated_delta_rule_fwd_kernel_h_blockdim64)
        h_pre = h_pre * np.exp(gc_last)[:, None] + kg.T @ v_new

    return o, h_pre


def run_check(T, D, Dv, chunk_size, seed, with_initial_state):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=(T, D)) * 0.3
    k = rng.normal(size=(T, D)) * 0.3
    v = rng.normal(size=(T, Dv)) * 0.3
    # raw per-token log-decay: keep decay close to but below 1 (typical trained regime)
    g_raw = -np.abs(rng.normal(size=(T, D))) * 0.05
    b = rng.uniform(0.2, 1.0, size=(T, D))     # erase gate in (0,1]-ish range
    w = rng.uniform(0.2, 1.0, size=(T, Dv))    # write gate
    scale = D ** -0.5
    h0 = rng.normal(size=(D, Dv)) * 0.1 if with_initial_state else None

    o_ref, h_ref = token_serial(q, k, v, g_raw, b, w, scale, h0=h0)
    o_wy, h_wy = chunked_wy(q, k, v, g_raw, b, w, scale, chunk_size, h0=h0)

    o_err = np.max(np.abs(o_ref - o_wy))
    h_err = np.max(np.abs(h_ref - h_wy))
    o_rel = o_err / (np.max(np.abs(o_ref)) + 1e-12)
    h_rel = h_err / (np.max(np.abs(h_ref)) + 1e-12)
    return o_err, o_rel, h_err, h_rel


if __name__ == "__main__":
    configs = [
        dict(T=64, D=16, Dv=16, chunk_size=16, seed=0, with_initial_state=False),
        dict(T=64, D=16, Dv=16, chunk_size=16, seed=1, with_initial_state=True),
        dict(T=128, D=32, Dv=24, chunk_size=32, seed=2, with_initial_state=True),
        dict(T=256, D=64, Dv=64, chunk_size=64, seed=3, with_initial_state=True),
        dict(T=192, D=48, Dv=32, chunk_size=64, seed=4, with_initial_state=False),  # 3 chunks
        dict(T=64, D=16, Dv=16, chunk_size=64, seed=5, with_initial_state=True),   # single chunk
    ]
    print(f"{'T':>5} {'D':>4} {'Dv':>4} {'C':>4}  {'o_abs_err':>12} {'o_rel_err':>12} {'h_abs_err':>12} {'h_rel_err':>12}")
    all_ok = True
    for cfg in configs:
        o_err, o_rel, h_err, h_rel = run_check(**cfg)
        ok = o_rel < 1e-9 and h_rel < 1e-9
        all_ok &= ok
        status = "OK" if ok else "MISMATCH"
        print(f"{cfg['T']:>5} {cfg['D']:>4} {cfg['Dv']:>4} {cfg['chunk_size']:>4}  "
              f"{o_err:>12.3e} {o_rel:>12.3e} {h_err:>12.3e} {h_rel:>12.3e}  {status}")
    print()
    print("ALL CONFIGS MATCH (float64, algebraic equivalence confirmed)" if all_ok else "MISMATCH DETECTED -- formula error")
