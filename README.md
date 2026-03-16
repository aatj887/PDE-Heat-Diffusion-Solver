# PDE Heat Diffusion Solver 🌡️

A Python application that simulates 1D heat diffusion using the Finite Difference Method (FTCS). The project includes a core NumPy-based solver and an interactive web dashboard built with Streamlit.

## 📖 Overview

This project solves the 1D Heat Equation:

$$ \frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2} $$

Where:
- $u(x, t)$: Temperature at position $x$ and time $t$ [°C]
- $\alpha$: Thermal diffusivity [m²/s]
- $x$: Position [m]
- $t$: Time [s]

### Numerical Method
We use the **Forward-Time Central-Space (FTCS)** explicit scheme to discretize the PDE:

$$ u_{i}^{n+1} = u_{i}^{n} + r \cdot (u_{i+1}^{n} - 2u_{i}^{n} + u_{i-1}^{n}) $$

Where $r = \frac{\alpha \Delta t}{(\Delta x)^2}$ is the diffusion number.

### Stability Condition
For the explicit method to remain stable, the diffusion number must satisfy the Courant-Friedrichs-Lewy (CFL) condition:

$$ r \leq 0.5 $$

The solver automatically checks this condition and raises an error if the time step is too large.

---

## 🚀 Features

- **Core Solver**: Pure NumPy implementation of the FTCS algorithm.
- **Vectorized**: Uses array slicing for high performance (no nested loops).
- **Interactive UI**: Real-time adjustment of parameters (Diffusivity, Grid Size, Time Step) via Streamlit.
- **Visualization**: Animated heat propagation plot using Plotly.

---

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/PDE-Heat-Diffusion-Solver.git
   cd PDE-Heat-Diffusion-Solver
   ```

2. **Install dependencies**
   Create a `requirements.txt` file with the following content:
   ```text
   numpy
   matplotlib
   streamlit
   plotly
   pandas
   ```
   Then install them:
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1. Running the Core Script
You can run the standalone solver script to generate a static Matplotlib plot:

```bash
python solver.py
```

### 2. Running the Interactive Dashboard
Launch the Streamlit app to visualize the heat diffusion dynamically:

```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
PDE-Heat-Diffusion-Solver/
│
├── solver.py        # Core logic: NumPy solver and Matplotlib static plotting
├── app.py           # Streamlit web application UI and Plotly animation
├── README.md        # Project documentation
└── requirements.txt # Python dependencies
```

---

## 🎮 App Controls (Streamlit)

The web application (`app.py`) allows you to tweak simulation parameters on the fly:

- **Diffusivity ($\alpha$)**: Controls how fast heat spreads. Higher values = faster diffusion.
- **Grid Size ($N_x$)**: The number of spatial points. Higher values increase spatial resolution but require more computation.
- **Time Step ($\Delta t$)**: The size of the time jump per iteration.
  > ⚠️ **Warning**: If $\Delta t$ is too large, the stability check will fail, and the app will ask you to decrease it.

---

## 📈 Example Output

**Initial Condition**: A "square pulse" of 100°C in the center of a 1-meter rod.
**Boundary Conditions**: Both ends held at 0°C (Dirichlet).

*The visualization shows the heat smoothing out over time until the entire rod reaches thermal equilibrium with the boundaries.*

---

## 🔭 Future Roadmap

- [ ] Implement Neumann (insulated) boundary conditions.
- [ ] Add 2D Heat Diffusion support.
- [ ] Allow users to draw custom initial temperature profiles.

