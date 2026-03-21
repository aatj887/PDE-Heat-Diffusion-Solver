# PDE Heat Diffusion Solver 🌡️

A Python application that solves the 1D heat equation using two methods: a classical Finite Difference solver and a trained Neural Network. Both methods are compared side-by-side in an interactive Streamlit dashboard.

## 📖 Overview

This project solves the 1D Heat Equation:

$$ \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2} $$

Where:
- $u(x, t)$: Temperature at position $x$ and time $t$ [°C]
- $\alpha$: Thermal diffusivity [m²/s]
- $x$: Position [m]
- $t$: Time [s]

---

## 🚀 Features

### Part 1 — Finite Difference Method (FTCS)
- Explicit **Forward-Time Central-Space (FTCS)** discretization:

$$ u_{i}^{n+1} = u_{i}^{n} + r \cdot (u_{i+1}^{n} - 2u_{i}^{n} + u_{i-1}^{n}), \quad r = \frac{\alpha \Delta t}{(\Delta x)^2} $$

- Automatic **CFL stability check** ($r \leq 0.5$) with user feedback.
- Interactive controls for diffusivity, grid size, time step, rod length, and total time.
- Execution time displayed in milliseconds.

### Part 2 — Neural Network Prediction
- A **4-layer MLP (256 units per layer)** trained to predict the final temperature profile directly from the initial condition and thermal diffusivity.
- Supports three initial condition shapes: square pulse, single Gaussian, two Gaussians.
- Optional overlay of the FD solver result for accuracy comparison (RMSE displayed).
- Inference time displayed in milliseconds.

### Speed Comparison
- Bar chart comparing FD solver vs NN execution time for the same problem.
- Speed-up factor shown as a metric.

---

## 📂 Project Structure

```
PDE-Heat-Diffusion-Solver/
│
├── solver.py           # Standalone FTCS solver with Matplotlib plot
├── generate_data.py    # Generates training data (CSV) by running many FD simulations
├── train_nn.py         # Trains the MLP and saves heat_nn.pt
├── app.py              # Streamlit dashboard (FD + NN + speed comparison)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/PDE-Heat-Diffusion-Solver.git
   cd PDE-Heat-Diffusion-Solver
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install torch            # required for the NN section
   ```

---

## 💻 Usage

### 1. Standalone FD solver
Generates a static Matplotlib plot of the heat diffusion:
```bash
python solver.py
```

### 2. Generate training data
Runs the FD solver across thousands of randomised initial conditions and thermal diffusivities, saving results to `heat_data.csv`:
```bash
python generate_data.py
```

Key settings at the top of the file:

| Variable | Default | Description |
|---|---|---|
| `N_SAMPLES` | 5 000 | Number of simulations |
| `N_SNAPSHOTS` | 20 | Time snapshots saved per simulation |
| `T_MAX` | 300 s | Maximum simulation time |
| `ALPHA_RANGE` | (1e-7, 1e-4) | Thermal diffusivity range |

### 3. Train the neural network
Loads `heat_data.csv`, trains the MLP, and saves `heat_nn.pt`:
```bash
python train_nn.py
```

The script prints train/val/test MSE and RMSE, and shows a training curve + prediction comparison plot.

### 4. Launch the interactive dashboard
```bash
streamlit run app.py
```

> **Note**: `heat_nn.pt` must exist before launching the app. Run steps 2 and 3 first.

---

## 🎮 Dashboard Controls

### FD Section
- **Diffusivity (α)**: Controls how fast heat spreads.
- **Grid Size (Nx)**: Number of spatial points (higher = finer resolution, slower).
- **Time Step (dt)**: Must satisfy the CFL condition; the app warns if unstable.
- **Rod Length / Total Time**: Set via number inputs.
- **▶️ / ⏸️ / 🔄**: Play, pause, and rewind the time animation.

### NN Section
- **Thermal diffusivity**: Log-spaced slider over the training range (1×10⁻⁷ to 1×10⁻⁴ m²/s).
- **Initial condition shape**: Square pulse, Gaussian, or two Gaussians with adjustable parameters.
- **Overlay FD ground truth**: Runs the FD solver on the same inputs and overlays it for comparison.

---

## 🧠 Neural Network Architecture

| Property | Value |
|---|---|
| Input | log₁₀(α) + 51 initial temperature values = 52 features |
| Hidden layers | 4 × 256 units (BatchNorm + ReLU + Dropout 0.1) |
| Output | 51 final temperature values |
| Loss | MSE |
| Optimiser | Adam (lr = 1×10⁻³) with ReduceLROnPlateau |
| Data split | 70% train / 15% val / 15% test |

Alpha is log-transformed before input because it spans three orders of magnitude. All inputs and outputs are standardised using training-set statistics stored in the checkpoint.

---

## 📈 Example Output

**Initial Condition**: A square pulse of 100°C in the centre of a 1-metre rod, with both ends held at 0°C (Dirichlet BCs).

The dashboard shows:
1. Temperature profile (initial → final)
2. Colour-coded rod visualisation (blue = cold, red = hot)
3. Full space-time heatmap (FD) or initial-vs-final comparison (NN)
4. Execution time bar chart comparing both methods
