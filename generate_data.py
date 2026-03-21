import numpy as np
import os

# ── Simulation constants (match solver.py) ──────────────────────────────────
L  = 1.0   # rod length [m]
nx = 51    # spatial points  (interior: 1..nx-2, boundaries fixed at 0)
dx = L / (nx - 1)
x  = np.linspace(0, L, nx)

# Thermal diffusivities for common materials [m²/s]
ALPHA_RANGE = (1e-7, 1e-4)   # covers insulators → metals

# We pick dt conservatively so the solver is stable for any alpha in range
alpha_max = ALPHA_RANGE[1]
dt = 0.4 * dx**2 / alpha_max   # r = alpha*dt/dx² ≤ 0.4  (< 0.5 stability limit)

T_MAX  = 300.0               # maximum simulation time [s]
nt_max = int(T_MAX / dt)

# How many time snapshots to record per simulation
N_SNAPSHOTS = 20

# ── Data-generation settings ─────────────────────────────────────────────────
N_SAMPLES    = 5_000   # total number of (IC, alpha) pairs to simulate
RANDOM_SEED  = 42
OUTPUT_FILE  = "heat_data.csv"


def make_initial_condition(rng: np.random.Generator) -> np.ndarray:
    """
    Return a temperature profile u0 of shape (nx,) with u0[0] = u0[-1] = 0.

    Randomly picks one of four IC families:
      1. Single square pulse
      2. Single Gaussian bump
      3. Sum of 1-3 Gaussian bumps
      4. Truncated random Fourier series (smooth IC)
    """
    u0 = np.zeros(nx)
    kind = rng.integers(0, 4)

    if kind == 0:
        # Square pulse: random position and width
        amp   = rng.uniform(20, 200)
        start = rng.uniform(0.05, 0.45)
        width = rng.uniform(0.05, 0.50 - start + 0.05)
        mask  = (x >= start) & (x <= start + width)
        u0[mask] = amp

    elif kind == 1:
        # Single Gaussian
        amp   = rng.uniform(20, 200)
        mu    = rng.uniform(0.15, 0.85)
        sigma = rng.uniform(0.03, 0.15)
        u0   += amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    elif kind == 2:
        # Superposition of 1–3 Gaussians
        n_bumps = rng.integers(1, 4)
        for _ in range(n_bumps):
            amp   = rng.uniform(10, 150)
            mu    = rng.uniform(0.1, 0.9)
            sigma = rng.uniform(0.02, 0.12)
            u0   += amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    else:
        # Truncated Fourier series (sum of sin modes satisfying BCs)
        n_modes = rng.integers(1, 6)
        for k in range(1, n_modes + 1):
            amp = rng.uniform(-100, 100)
            u0 += amp * np.sin(k * np.pi * x / L)

    # Enforce Dirichlet BCs and clip to a reasonable temperature range
    u0[0]  = 0.0
    u0[-1] = 0.0
    u0     = np.clip(u0, -300, 300)
    return u0


def run_solver(u0: np.ndarray, alpha: float, snapshot_steps: np.ndarray) -> np.ndarray:
    """
    Run FTCS finite-difference solver from initial condition u0.

    Parameters
    ----------
    u0             : initial temperature profile, shape (nx,)
    alpha          : thermal diffusivity
    snapshot_steps : sorted array of time-step indices at which to record u

    Returns
    -------
    snapshots : shape (len(snapshot_steps), nx)
    """
    r = alpha * dt / dx**2   # stability parameter  (≤ 0.4 by construction)
    u = u0.copy()
    snapshots = np.empty((len(snapshot_steps), nx))
    snap_idx  = 0

    for n in range(snapshot_steps[-1] + 1):
        if n == snapshot_steps[snap_idx]:
            snapshots[snap_idx] = u
            snap_idx += 1
            if snap_idx == len(snapshot_steps):
                break
        # FTCS update (interior only; boundaries stay 0)
        u[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])

    return snapshots


def generate_dataset(n_samples: int, seed: int) -> dict:
    """
    Generate the full dataset.

    Each (IC, alpha) pair is simulated once; N_SNAPSHOTS time levels are saved.

    Returned arrays
    ---------------
    u0        : (n_samples, nx)            – initial conditions
    alpha_arr : (n_samples,)               – thermal diffusivities
    t_arr     : (n_samples, N_SNAPSHOTS)   – simulation times for each snapshot
    u_t       : (n_samples, N_SNAPSHOTS, nx) – temperature profiles
    x         : (nx,)                      – spatial grid
    """
    rng = np.random.default_rng(seed)

    u0_arr    = np.empty((n_samples, nx),              dtype=np.float32)
    alpha_arr = np.empty((n_samples,),                 dtype=np.float32)
    t_arr     = np.empty((n_samples, N_SNAPSHOTS),     dtype=np.float32)
    u_t_arr   = np.empty((n_samples, N_SNAPSHOTS, nx), dtype=np.float32)

    # Pre-build a pool of snapshot step indices (same for all samples for speed)
    # We exclude t=0 from the snapshots (that's u0 itself)
    snapshot_steps = np.unique(
        np.linspace(1, nt_max, N_SNAPSHOTS, dtype=int)
    )

    for i in range(n_samples):
        u0    = make_initial_condition(rng)
        alpha = rng.uniform(*ALPHA_RANGE)

        snapshots = run_solver(u0, alpha, snapshot_steps)
        times     = snapshot_steps * dt

        u0_arr[i]    = u0.astype(np.float32)
        alpha_arr[i] = alpha
        t_arr[i]     = times.astype(np.float32)
        u_t_arr[i]   = snapshots.astype(np.float32)

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1:,} / {n_samples:,} samples")

    return {
        "u0":    u0_arr,
        "alpha": alpha_arr,
        "t":     t_arr,
        "u_t":   u_t_arr,
        "x":     x.astype(np.float32),
        "dt":    np.float32(dt),
    }


if __name__ == "__main__":
    print(f"Simulation grid  : nx={nx}, dx={dx:.5f} m")
    print(f"Time step (dt)   : {dt:.6f} s  (stable for α ≤ {alpha_max:.2e})")
    print(f"Max sim time     : {T_MAX} s  ({nt_max} steps)")
    print(f"Snapshots/sample : {N_SNAPSHOTS}")
    print(f"Generating {N_SAMPLES:,} samples …\n")

    data = generate_dataset(N_SAMPLES, RANDOM_SEED)

    # ── Flatten to 2-D: one row per (sample, snapshot) ───────────────────────
    # Columns: sample_id | alpha | t | u0_0..u0_50 | u_0..u_50
    u0_cols  = [f"u0_{i}" for i in range(nx)]
    u_t_cols = [f"u_{i}"  for i in range(nx)]

    n_samples   = data["u0"].shape[0]
    n_snapshots = data["t"].shape[1]
    rows = []
    for i in range(n_samples):
        for s in range(n_snapshots):
            row = [i, data["alpha"][i], data["t"][i, s]]
            row.extend(data["u0"][i].tolist())
            row.extend(data["u_t"][i, s].tolist())
            rows.append(row)

    columns = ["sample_id", "alpha", "t"] + u0_cols + u_t_cols
    import csv
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    size_mb = os.path.getsize(OUTPUT_FILE) / 1024**2
    print(f"\nSaved '{OUTPUT_FILE}'  ({size_mb:.1f} MB)")
    print(f"  Rows: {len(rows):,}  (samples × snapshots = {n_samples} × {n_snapshots})")
    print(f"  Columns: {len(columns)}  (sample_id, alpha, t, {nx} u0 values, {nx} u_t values)")

    print("\nSample 0 stats:")
    print(f"  alpha = {data['alpha'][0]:.4e}")
    print(f"  u0    max={data['u0'][0].max():.2f}  min={data['u0'][0].min():.2f}")
    print(f"  u_t   max={data['u_t'][0].max():.2f}  (final snapshot, should be smaller)")
