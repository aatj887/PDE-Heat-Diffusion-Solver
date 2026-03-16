import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io
import time

# Set page config
st.set_page_config(
    page_title="Heat Diffusion Solver",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔥 1D Heat Diffusion Solver")
st.markdown("Interactive visualization of heat diffusion through a 1D rod")

# Initialize session state for simulation control
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'solver_params' not in st.session_state:
    st.session_state.solver_params = None

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

            # Solve
            for n in range(0, nt):
                u[n+1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])

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

        # Create figure with multiple subplots
        fig = plt.figure(figsize=(12, 8))

        # 1. Temperature curve plot (top)
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(x, u[st.session_state.current_step, :], 'b-', linewidth=2, label=f"t = {current_time:.2f}s")
        ax1.plot(x, u[0, :], 'g--', alpha=0.5, label="Initial (t=0)")
        if nt > 0:
            ax1.plot(x, u[-1, :], 'r--', alpha=0.5, label=f"Final (t={T:.2f}s)")
        ax1.set_ylabel("Temperature (°C)", fontsize=11)
        ax1.set_title("1D Heat Diffusion through Rod", fontsize=13, fontweight='bold')
        ax1.set_ylim([-5, 105])
        ax1.legend(fontsize=9, loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. Colored rod visualization (middle) - showing actual rod with color gradient
        ax2 = plt.subplot(3, 1, 2)

        # Create a color-coded representation of the rod
        current_temps = u[st.session_state.current_step, :]

        # Normalize temperatures to [0, 1] for color mapping (0°C = blue, 100°C = orange/red)
        normalized_temps = np.clip(current_temps, 0, 100) / 100.0

        # Create colored segments along the rod
        segment_width = L / (nx - 1)
        for i in range(nx - 1):
            x_start = x[i]
            x_end = x[i + 1]
            temp_avg = (current_temps[i] + current_temps[i + 1]) / 2
            normalized = np.clip(temp_avg, 0, 100) / 100.0

            # Create gradient: blue (cold) -> orange/red (hot)
            # Blue: [0.0, 0.0, 1.0] -> Orange: [1.0, 0.65, 0.0] -> Red: [1.0, 0.0, 0.0]
            if normalized < 0.5:
                # Transition from blue to orange
                r = 2 * normalized
                g = 0.65 * 2 * normalized
                b = 1.0 - 2 * normalized
            else:
                # Transition from orange to red
                r = 1.0
                g = 0.65 * (1.0 - (normalized - 0.5) * 2)
                b = 0.0

            ax2.barh(0, segment_width, left=x_start, height=0.5,
                    color=(r, g, b), edgecolor='black', linewidth=0.5)

        ax2.set_ylim(-0.5, 0.5)
        ax2.set_xlim(0, L)
        ax2.set_xlabel("Position along rod (m)", fontsize=11)
        ax2.set_yticks([])
        ax2.set_title("Rod Temperature Distribution (Color: Blue=Cold → Orange/Red=Hot)", fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.2, axis='x')

        # 3. 2D Heatmap showing evolution over time (bottom)
        ax3 = plt.subplot(3, 1, 3)

        heatmap_data = u[:st.session_state.current_step + 1, :]
        times = np.arange(0, st.session_state.current_step + 1) * delta_t

        if heatmap_data.shape[0] >= 2:
            im = ax3.contourf(x, times, heatmap_data, levels=20, cmap='coolwarm')

            ax3.axhline(
                y=current_time,
                color='white',
                linestyle='--',
                linewidth=2,
                label=f'Current: {current_time:.2f}s'
            )

            plt.colorbar(im, ax=ax3, label='Temperature (°C)')

        else:
            ax3.text(
                0.5,
                0.5,
                "Heatmap will appear after the first timestep",
                ha="center",
                va="center",
                transform=ax3.transAxes
            )

        ax3.set_xlabel("Position along rod (m)", fontsize=11)
        ax3.set_ylabel("Time (s)", fontsize=11)
        ax3.set_title("Temperature Evolution Over Time", fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)


        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Statistics
        st.subheader("📈 Statistics")
        col_stat1, col_stat2, col_stat3 = st.columns(3)

        with col_stat1:
            max_temp = np.max(u[st.session_state.current_step, :])
            st.metric("Max Temperature", f"{max_temp:.2f}°C")

        with col_stat2:
            mean_temp = np.mean(u[st.session_state.current_step, :])
            st.metric("Mean Temperature", f"{mean_temp:.2f}°C")

        with col_stat3:
            min_temp = np.min(u[st.session_state.current_step, :])
            st.metric("Min Temperature", f"{min_temp:.2f}°C")

    elif not is_stable:
        st.warning("⚠️ Simulation parameters are unstable. Adjust the parameters to continue.")

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
