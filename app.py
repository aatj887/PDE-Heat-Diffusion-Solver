import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import time
import os

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ── Shared chart helper ───────────────────────────────────────────────────────
def plot_heat_charts(x, u0, u_final, label, T, L=1.0, u_all=None, dt=None):
    """
    3-panel figure used by both the FD and NN sections.

    Panel 1 – temperature curve (initial + final)
    Panel 2 – colour-coded rod of the final state
    Panel 3 – FD: full contourf heatmap over time
               NN: 2-row imshow (initial row / final row)
    """
    fig = plt.figure(figsize=(12, 8))

    # 1. Temperature curve
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(x, u0,      'g--', alpha=0.7, lw=1.5, label="Initial (t = 0)")
    ax1.plot(x, u_final, 'r-',  lw=2,      label=f"Final (t = {T:.0f} s)")
    ax1.set_ylabel("Temperature (°C)", fontsize=11)
    ax1.set_title(f"1D Heat Diffusion — {label}", fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # 2. Colour-coded rod (final state)
    ax2 = plt.subplot(3, 1, 2)
    seg_w = L / (len(x) - 1)
    for i in range(len(x) - 1):
        temp_avg = (u_final[i] + u_final[i + 1]) / 2.0
        norm = np.clip(temp_avg, 0, 100) / 100.0
        if norm < 0.5:
            rc = 2 * norm; gc = 0.65 * 2 * norm; bc = 1.0 - 2 * norm
        else:
            rc = 1.0; gc = 0.65 * (1.0 - (norm - 0.5) * 2); bc = 0.0
        ax2.barh(0, seg_w, left=x[i], height=0.5,
                 color=(rc, gc, bc), edgecolor='black', linewidth=0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.set_xlim(0, L)
    ax2.set_xlabel("Position along rod (m)", fontsize=11)
    ax2.set_yticks([])
    ax2.set_title("Final Rod Temperature Distribution (Blue=Cold → Red=Hot)",
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.2, axis='x')

    # 3. Heatmap
    ax3 = plt.subplot(3, 1, 3)
    if u_all is not None and dt is not None:
        times = np.arange(u_all.shape[0]) * dt
        im = ax3.contourf(x, times, u_all, levels=20, cmap='coolwarm')
        plt.colorbar(im, ax=ax3, label='Temperature (°C)')
        ax3.set_ylabel("Time (s)", fontsize=11)
        ax3.set_title("Temperature Evolution Over Time", fontsize=12, fontweight='bold')
    else:
        two_rows = np.vstack([u0.reshape(1, -1), u_final.reshape(1, -1)])
        im = ax3.imshow(two_rows, aspect='auto', cmap='coolwarm',
                        extent=[0, L, 0, 1], origin='lower')
        plt.colorbar(im, ax=ax3, label='Temperature (°C)')
        ax3.set_yticks([0.25, 0.75])
        ax3.set_yticklabels(["t = 0", f"t = {T:.0f} s"])
        ax3.set_title("Initial vs Final Temperature", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Position along rod (m)", fontsize=11)

    plt.tight_layout()
    return fig


# Set page config
st.set_page_config(
    page_title="Heat Diffusion Solver",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔥 1D Heat Diffusion Solver")
st.markdown("Interactive visualization of heat diffusion through a 1D rod")

st.divider()
st.subheader("📐 Part 1 — Finite Difference Method (FTCS)")
st.markdown(
    "The simulation below solves the 1D heat equation numerically using the "
    "**Forward-Time Centered-Space (FTCS)** explicit finite difference scheme."
)

# Initialize session state for simulation control
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'solver_params' not in st.session_state:
    st.session_state.solver_params = None
if 'fd_solve_time' not in st.session_state:
    st.session_state.fd_solve_time = None
if 'nn_infer_time' not in st.session_state:
    st.session_state.nn_infer_time = None
if 'nn_fd_compare_time' not in st.session_state:
    st.session_state.nn_fd_compare_time = None

# Create two columns for layout
col1, col2 = st.columns([1, 2])

# ============= LEFT COLUMN: CONTROLS =============
with col1:
    st.subheader("⚙️ Parameters")

    # Sliders for parameters
    alpha = st.slider(
        "Diffusivity (α)",
        min_value=1e-6,
        max_value=1e-4,
        value=1.172e-5,
        format="%.6e",
        help="Thermal diffusivity in m²/s"
    )

    nx = st.slider(
        "Grid Size (Nx)",
        min_value=11,
        max_value=201,
        value=51,
        step=10,
        help="Number of spatial points"
    )

    delta_t = st.slider(
        "Time Step (dt)",
        min_value=0.01,
        max_value=1.0,
        value=0.1,
        step=0.01,
        help="Time step in seconds"
    )

    # Additional parameters (non-sliders)
    st.subheader("Advanced Settings")
    L = st.number_input("Rod Length (m)", min_value=0.1, value=1.0, step=0.1)
    T = st.number_input("Total Time (s)", min_value=10, value=600, step=10)

    # Stability check
    delta_x = L / (nx - 1)
    max_dt = 0.5 * delta_x**2 / alpha
    is_stable = delta_t <= max_dt

    if is_stable:
        st.success(f"✓ Stable: dt ({delta_t:.4f}) ≤ max dt ({max_dt:.4f})")
    else:
        st.error(f"✗ Unstable: dt ({delta_t:.4f}) > max dt ({max_dt:.4f})")
        st.warning("Please reduce dt or increase grid size for stability!")

    # Check if parameters changed
    current_params = (alpha, nx, delta_t, L, T)
    if st.session_state.solver_params != current_params:
        st.session_state.current_step = 0
        st.session_state.is_running = False
        st.session_state.solver_params = current_params
        st.session_state.simulation_data = None
        st.session_state.fd_solve_time = None

# ============= RIGHT COLUMN: VISUALIZATION =============
with col2:
    st.subheader("📊 Simulation Visualization")

    # Solve the heat diffusion if needed or if parameters changed
    if st.session_state.simulation_data is None and is_stable:
        with st.spinner("Solving heat diffusion equation..."):
            # Discretization
            delta_x = L / (nx - 1)
            nt = int(T / delta_t)

            # Create grid
            u = np.zeros((nt + 1, nx))

            # Initial conditions: square pulse in the middle
            u[0, int(nx/3) : int(2*nx/3)] = 100.0

            # Boundary conditions
            u[:, 0] = 0.0
            u[:, -1] = 0.0

            # Stability constant
            r = alpha * delta_t / delta_x**2

            # Solve (timed)
            _fd_t0 = time.perf_counter()
            for n in range(0, nt):
                u[n+1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
            st.session_state.fd_solve_time = time.perf_counter() - _fd_t0

            st.session_state.simulation_data = u
            st.session_state.current_step = 0

    if st.session_state.simulation_data is not None and is_stable:
        u = st.session_state.simulation_data
        nt = u.shape[0] - 1
        x = np.linspace(0, L, nx)

        # Simulation controls
        col_play, col_pause, col_rewind = st.columns(3)

        with col_play:
            if st.button("▶️ Start", use_container_width=True):
                st.session_state.is_running = True

        with col_pause:
            if st.button("⏸️ Pause", use_container_width=True):
                st.session_state.is_running = False

        with col_rewind:
            if st.button("🔄 Rewind", use_container_width=True):
                st.session_state.current_step = 0
                st.session_state.is_running = False

        # Time slider for manual control
        current_step = st.slider(
            "Time Step",
            min_value=0,
            max_value=nt,
            value=st.session_state.current_step,
            step=1
        )
        st.session_state.current_step = current_step

        if st.session_state.is_running:
            if st.session_state.current_step < nt:
                st.session_state.current_step += 1
                time.sleep(0.05)   # controls animation speed
                st.rerun()
            else:
                st.session_state.is_running = False

        # Display current time
        current_time = st.session_state.current_step * delta_t
        st.metric("Current Time (s)", f"{current_time:.2f}", f"of {T:.2f}s")

        # Statistics
        st.subheader("📈 Statistics")
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)

        with col_stat1:
            max_temp = np.max(u[st.session_state.current_step, :])
            st.metric("Max Temperature", f"{max_temp:.2f}°C")

        with col_stat2:
            mean_temp = np.mean(u[st.session_state.current_step, :])
            st.metric("Mean Temperature", f"{mean_temp:.2f}°C")

        with col_stat3:
            min_temp = np.min(u[st.session_state.current_step, :])
            st.metric("Min Temperature", f"{min_temp:.2f}°C")

        with col_stat4:
            if st.session_state.fd_solve_time is not None:
                st.metric("FD Solve Time", f"{st.session_state.fd_solve_time * 1e3:.1f} ms")

        # Final-state charts (same layout as NN section)
        st.markdown("**Final State**")
        fig_fd_final = plot_heat_charts(
            x, u[0], u[-1], "Finite Difference (FTCS)", T,
            L=L, u_all=u, dt=delta_t,
        )
        st.pyplot(fig_fd_final)
        plt.close(fig_fd_final)

    elif not is_stable:
        st.warning("⚠️ Simulation parameters are unstable. Adjust the parameters to continue.")

# ============= NEURAL NETWORK SECTION =============
st.divider()
st.subheader("🧠 Neural Network Prediction")

NN_NX       = 51          # grid size the model was trained on
NN_T_MAX    = 300.0       # simulation duration the model was trained on
NN_L        = 1.0
NN_ALPHA_MAX = 1e-4
NN_DX       = NN_L / (NN_NX - 1)
NN_DT       = 0.4 * NN_DX**2 / NN_ALPHA_MAX   # stable dt used during training
NN_NT       = int(NN_T_MAX / NN_DT)
NN_X        = np.linspace(0, NN_L, NN_NX)

MODEL_FILE  = "heat_nn.pt"

# ── Hidden class (must match train_nn.py) ──
if TORCH_AVAILABLE:
    class HeatMLP(nn.Module):
        def __init__(self, n_in, n_out, hidden=256, n_layers=4):
            super().__init__()
            layers = []
            in_dim = n_in
            for _ in range(n_layers):
                layers += [nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden),
                           nn.ReLU(), nn.Dropout(p=0.1)]
                in_dim = hidden
            layers.append(nn.Linear(hidden, n_out))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    @st.cache_resource
    def load_model(path):
        ckpt  = torch.load(path, map_location="cpu", weights_only=False)
        model = HeatMLP(ckpt["n_in"], ckpt["n_out"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt

    def nn_predict(model, ckpt, alpha_val, u0):
        X_mean, X_std = ckpt["X_mean"], ckpt["X_std"]
        y_mean, y_std = ckpt["y_mean"], ckpt["y_std"]
        log_alpha = np.array([[np.log10(alpha_val)]], dtype=np.float32)
        u0_row    = u0.reshape(1, -1).astype(np.float32)
        X = np.hstack([log_alpha, u0_row])
        X_norm = ((X - X_mean) / X_std).astype(np.float32)
        with torch.no_grad():
            y_norm = model(torch.from_numpy(X_norm)).numpy()
        return (y_norm * y_std + y_mean).flatten()

    def run_fd_final(u0, alpha_val):
        """FD solver returning only the final-time profile (matches training setup)."""
        r = alpha_val * NN_DT / NN_DX**2
        u = u0.copy()
        for _ in range(NN_NT):
            u[1:-1] = u[1:-1] + r * (u[2:] - 2 * u[1:-1] + u[:-2])
        return u

if not TORCH_AVAILABLE:
    st.warning("PyTorch is not installed. Run `pip install torch` to enable NN predictions.")
elif not os.path.exists(MODEL_FILE):
    st.warning(f"Model file `{MODEL_FILE}` not found. Run `train_nn.py` first to generate it.")
else:
    model, ckpt = load_model(MODEL_FILE)

    nn_col1, nn_col2 = st.columns([1, 2])

    with nn_col1:
        st.markdown("**Parameters**")

        nn_alpha = st.select_slider(
            "Thermal diffusivity α (m²/s)",
            options=[round(v, 9) for v in np.geomspace(1e-7, 1e-4, 40)],
            value=round(float(np.geomspace(1e-7, 1e-4, 40)[19]), 9),
            format_func=lambda v: f"{v:.2e}",
            help="Range used during training: 1×10⁻⁷ to 1×10⁻⁴ m²/s",
        )

        st.markdown("**Initial condition**")
        ic_type = st.selectbox(
            "Shape",
            ["Square pulse", "Gaussian", "Two Gaussians"],
        )

        if ic_type == "Square pulse":
            amp_sq   = st.slider("Amplitude (°C)",    10.0, 200.0, 100.0, 5.0)
            start_sq = st.slider("Start position (m)", 0.05, 0.45,  0.25, 0.05)
            width_sq = st.slider("Width (m)",           0.05, 0.50,  0.25, 0.05)
            nn_u0 = np.zeros(NN_NX)
            mask  = (NN_X >= start_sq) & (NN_X <= start_sq + width_sq)
            nn_u0[mask] = amp_sq

        elif ic_type == "Gaussian":
            amp_g   = st.slider("Amplitude (°C)",  10.0, 200.0, 100.0, 5.0)
            mu_g    = st.slider("Centre (m)",        0.1,  0.9,   0.5,  0.05)
            sigma_g = st.slider("Spread σ (m)",     0.02,  0.20,  0.08, 0.01)
            nn_u0   = amp_g * np.exp(-0.5 * ((NN_X - mu_g) / sigma_g) ** 2)

        else:  # Two Gaussians
            amp1  = st.slider("Amplitude 1 (°C)", 10.0, 200.0, 80.0,  5.0)
            mu1   = st.slider("Centre 1 (m)",       0.1,  0.5,   0.25, 0.05)
            amp2  = st.slider("Amplitude 2 (°C)", 10.0, 200.0, 120.0, 5.0)
            mu2   = st.slider("Centre 2 (m)",       0.5,  0.9,   0.75, 0.05)
            sig   = st.slider("Spread σ (m)",      0.02,  0.20,  0.08, 0.01)
            nn_u0 = (amp1 * np.exp(-0.5 * ((NN_X - mu1) / sig) ** 2) +
                     amp2 * np.exp(-0.5 * ((NN_X - mu2) / sig) ** 2))

        nn_u0[0]  = 0.0   # enforce Dirichlet BCs
        nn_u0[-1] = 0.0

        run_prediction = st.button("🔮 Predict final state", use_container_width=True)
        show_fd = st.checkbox("Overlay FD solver ground truth", value=True,
                              help="Runs the finite-difference solver for comparison. "
                                   "Slow for small α.")

    with nn_col2:
        fig_nn, ax_nn = plt.subplots(figsize=(9, 4))
        ax_nn.plot(NN_X, nn_u0, color="steelblue", lw=2, label="Initial condition (t = 0)")
        ax_nn.set_xlabel("Position (m)")
        ax_nn.set_ylabel("Temperature (°C)")
        ax_nn.set_title("Initial condition preview")
        ax_nn.grid(True, alpha=0.3)
        ax_nn.legend()
        st.pyplot(fig_nn)
        plt.close(fig_nn)

        if run_prediction:
            with st.spinner("Running NN inference …"):
                _nn_t0 = time.perf_counter()
                nn_pred = nn_predict(model, ckpt, nn_alpha, nn_u0)
                st.session_state.nn_infer_time = time.perf_counter() - _nn_t0

            fd_final = None
            if show_fd:
                with st.spinner("Running FD solver for comparison …"):
                    _fd_t0 = time.perf_counter()
                    fd_final = run_fd_final(nn_u0.copy(), nn_alpha)
                    st.session_state.nn_fd_compare_time = time.perf_counter() - _fd_t0

            # Same 3-panel layout as the FD section (no full u_all, so panel 3 → imshow)
            fig_nn_pred = plot_heat_charts(
                NN_X, nn_u0, nn_pred, "Neural Network", NN_T_MAX, L=NN_L,
            )
            # Overlay FD comparison on panel 1
            if fd_final is not None:
                fig_nn_pred.axes[0].plot(
                    NN_X, fd_final, color="seagreen", lw=1.5, ls=":",
                    label=f"FD solver (t = {NN_T_MAX:.0f} s)",
                )
                fig_nn_pred.axes[0].legend(fontsize=9, loc='upper right')
                rmse = float(np.sqrt(np.mean((nn_pred - fd_final) ** 2)))
                fig_nn_pred.axes[0].set_title(
                    f"1D Heat Diffusion — Neural Network  |  RMSE vs FD = {rmse:.3f} °C",
                    fontsize=13, fontweight='bold',
                )
            st.pyplot(fig_nn_pred)
            plt.close(fig_nn_pred)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("NN peak temperature",  f"{nn_pred.max():.2f} °C")
            c2.metric("NN mean temperature",  f"{nn_pred.mean():.2f} °C")
            c3.metric("NN inference time",    f"{st.session_state.nn_infer_time * 1e3:.2f} ms")
            if fd_final is not None:
                c4.metric("RMSE vs FD solver", f"{rmse:.4f} °C")

# ============= SPEED COMPARISON =============
st.divider()
st.subheader("⚡ Speed Comparison")

_fd_t  = st.session_state.fd_solve_time
_nn_t  = st.session_state.nn_infer_time
_fdc_t = st.session_state.nn_fd_compare_time

if _fd_t is None or _nn_t is None:
    st.info(
        "Run the FD solver (Part 1) and press **Predict** in the NN section "
        "to see the speed comparison."
    )
else:
    # The FD solver in Part 1 uses user-chosen parameters (grid size, dt, T).
    # The NN comparison FD run uses fixed training parameters (nx=51, T=300s).
    # We compare the fixed-parameter FD run vs NN inference for a fair apples-to-apples race.
    fd_time_ms = (_fdc_t if _fdc_t is not None else _fd_t) * 1e3
    nn_time_ms = _nn_t * 1e3

    speedup = fd_time_ms / nn_time_ms if nn_time_ms > 0 else float('inf')

    m1, m2, m3 = st.columns(3)
    m1.metric("FD solver time",    f"{fd_time_ms:.1f} ms")
    m2.metric("NN inference time", f"{nn_time_ms:.2f} ms")
    m3.metric("Speed-up (FD / NN)", f"{speedup:.1f}×")

    fig_bar, ax_bar = plt.subplots(figsize=(5, 3))
    methods = ["FD Solver", "Neural Network"]
    times   = [fd_time_ms, nn_time_ms]
    colors  = ["steelblue", "tomato"]
    bars = ax_bar.bar(methods, times, color=colors, width=0.4)
    for bar, t in zip(bars, times):
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.02,
            f"{t:.2f} ms", ha='center', va='bottom', fontsize=10, fontweight='bold',
        )
    ax_bar.set_ylabel("Execution time (ms)")
    ax_bar.set_title("FD Solver vs Neural Network — Execution Time")
    ax_bar.set_ylim(0, max(times) * 1.25)
    ax_bar.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_bar)
    plt.close(fig_bar)

    st.caption(
        "FD time shown is for the fixed training-equivalent run (nx=51, T=300 s). "
        "The NN time is for a single forward pass on CPU."
    )

# ============= VIDEO SECTION =============
st.divider()
st.subheader("📹 Video Tutorial")

with st.expander("▶️ Watch an embedded video", expanded=False):
    st.markdown("### Heat Diffusion Explanation")

    # Replace 'dQw4w9WgXcQ' with your actual unlisted YouTube video ID
    video_id = st.text_input(
        "Enter YouTube Video ID:",
        value="",
        placeholder="e.g., dQw4w9WgXcQ (for unlisted videos, use the full ID)"
    )

    if video_id:
        # Embed YouTube video
        st.video(f"https://www.youtube.com/watch?v={video_id}")
    else:
        st.info("👆 Enter a YouTube video ID above to embed an unlisted video")

# ============= INFORMATION SECTION =============
st.divider()
st.subheader("ℹ️ About This Solver")

with st.expander("How does it work?"):
    st.markdown("""
    ### 1D Heat Diffusion Equation

    This solver uses the **finite difference method** to solve the 1D heat diffusion equation:

    ∂u/∂t = α ∂²u/∂x²

    Where:
    - **u** = temperature
    - **α** = thermal diffusivity (m²/s)
    - **t** = time
    - **x** = position along the rod

    ### Numerical Method
    - **Discretization**: Forward in time, centered in space (FTCS)
    - **Stability Condition**: Δt ≤ 0.5 × (Δx)² / α

    ### Initial Condition
    - A "square pulse" of heat (100°C) in the middle third of the rod
    - The ends are held at 0°C (Dirichlet boundary condition)

    ### Visualization
    - Blue line: Current temperature distribution
    - Green dashed: Initial condition
    - Red dashed: Final state
    """)

with st.expander("How to use the controls"):
    st.markdown("""
    ### Parameter Controls
    - **Diffusivity (α)**: Higher values = faster heat spreading
    - **Grid Size (Nx)**: More points = higher spatial resolution (slower)
    - **Time Step (dt)**: Smaller values = better accuracy (slower)

    ### Simulation Controls
    - **▶️ Start**: Begin automatic playback through time steps
    - **⏸️ Pause**: Stop the playback
    - **🔄 Rewind**: Return to the beginning
    - **Time Step Slider**: Manually navigate through time

    ### Stability
    The solver checks the stability condition automatically. If unstable, reduce dt or increase grid size.
    """)
