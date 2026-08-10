"""
Run this in your actual JAX/TPU (or CPU) environment -- the sandbox used to
derive gdn2_wy_reference.py has no network access to install JAX, so the
formulas were verified in NumPy (see verify_wy_numpy.py, same algebra,
matched the token-serial ground truth to ~1e-15 relative error across 6
configs). This script re-runs the same style of check directly in JAX so you
can confirm bit-for-bit-reasonable agreement (float32) before we move to
Milestone 2 (Pallas).

Usage:
    python test_gdn2_wy_reference.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from gdn2_wy_reference import gdn2_token_serial_reference, gdn2_chunked_wy_reference


def run_check(key, B, L, H, D, Dv, chunk_size, with_initial_state, dtype=jnp.float32):
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    q = jax.random.normal(k1, (B, L, H, D), dtype=dtype) * 0.3
    k = jax.random.normal(k2, (B, L, H, D), dtype=dtype) * 0.3
    v = jax.random.normal(k3, (B, L, H, Dv), dtype=dtype) * 0.3
    g = -jnp.abs(jax.random.normal(k4, (B, L, H, D), dtype=dtype)) * 0.05
    b = jax.random.uniform(k5, (B, L, H, D), minval=0.2, maxval=1.0, dtype=dtype)
    w = jax.random.uniform(k6, (B, L, H, Dv), minval=0.2, maxval=1.0, dtype=dtype)
    scale = D ** -0.5
    h0 = jax.random.normal(k7, (B, H, D, Dv), dtype=dtype) * 0.1 if with_initial_state else None

    o_ref, h_ref = gdn2_token_serial_reference(q, k, v, g, b, w, scale, h0=h0)
    o_wy, h_wy = gdn2_chunked_wy_reference(q, k, v, g, b, w, scale, chunk_size, h0=h0)

    o_ref, h_ref, o_wy, h_wy = map(lambda x: np.asarray(x, dtype=np.float64), (o_ref, h_ref, o_wy, h_wy))

    o_err = np.max(np.abs(o_ref - o_wy))
    h_err = np.max(np.abs(h_ref - h_wy))
    o_rel = o_err / (np.max(np.abs(o_ref)) + 1e-12)
    h_rel = h_err / (np.max(np.abs(h_ref)) + 1e-12)
    return o_err, o_rel, h_err, h_rel


if __name__ == "__main__":
    configs = [
        dict(B=1, L=64, H=1, D=16, Dv=16, chunk_size=16, with_initial_state=False),
        dict(B=2, L=64, H=2, D=16, Dv=16, chunk_size=16, with_initial_state=True),
        dict(B=1, L=128, H=4, D=32, Dv=32, chunk_size=32, with_initial_state=True),
        dict(B=1, L=256, H=8, D=64, Dv=64, chunk_size=64, with_initial_state=True),  # d_head=64 like model.py's n_heads=8
        dict(B=1, L=192, H=1, D=48, Dv=32, chunk_size=64, with_initial_state=False),  # 3 chunks
        dict(B=1, L=64, H=1, D=16, Dv=16, chunk_size=64, with_initial_state=True),   # single chunk
    ]
    key = jax.random.PRNGKey(0)
    print(f"{'B':>3} {'L':>5} {'H':>3} {'D':>4} {'Dv':>4} {'C':>4}  {'o_rel_err':>12} {'h_rel_err':>12}")
    all_ok = True
    for i, cfg in enumerate(configs):
        subkey = jax.random.fold_in(key, i)
        o_err, o_rel, h_err, h_rel = run_check(subkey, **cfg)
        ok = o_rel < 1e-3 and h_rel < 1e-3  # float32 tolerance (numpy fp64 proof already gave ~1e-15)
        all_ok &= ok
        status = "OK" if ok else "MISMATCH"
        print(f"{cfg['B']:>3} {cfg['L']:>5} {cfg['H']:>3} {cfg['D']:>4} {cfg['Dv']:>4} {cfg['chunk_size']:>4}  "
              f"{o_rel:>12.3e} {h_rel:>12.3e}  {status}")
    print()
    print("ALL CONFIGS MATCH (float32 JAX)" if all_ok else "MISMATCH -- check before proceeding to Milestone 2")
