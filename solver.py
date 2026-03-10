import numpy as np
import matplotlib.pyplot as plt

## --- 0. Initial parameters ---
L = 1 # Length of the rod in metres
T = 600 # Total time to simulate in seconds
alpha = 0.00001172 # Thermal diffusivity in square metres per second (for Steel)
nx = 51 # Number of spatial points


## --- 1. Discretization ---
delta_x = L / (nx - 1) # Distance between spatial points

# We need to calculate delta_t as we need to satisfy the stability condition
delta_t = 0.1 # Distance between snapshots, in seconds

if delta_t <= 0.5 * delta_x**2 / alpha:
    print(f'A delta_t of {delta_t} satisfies our stability condition')
else:
    raise ValueError('The chosen delta_t will break our model. Please choose a smaller delta_t')

nt = int(T / delta_t) # Number of time steps

print(f"Grid Spacing (dx): {delta_x:.5f} [m]")
print(f"Time Step (dt):    {delta_t:.5f} [s]")
print(f"Total Time Steps:  {nt}")

## --- 2. The Grid Array ---
# We create a matrix: Rows = Time steps, Columns = Space points
# We initialize it with zeros (the rod starts at 0 degrees)
u = np.zeros((nt + 1, nx))

# --- 3. Initial Conditions ---
# This represents a "square pulse" heat in the middle.
# The ends of the rod are 0 degrees, but the middle section is 100 degrees.
# We modify the FIRST row (time t=0)
u[0, int(nx/3) : int(2*nx/3)] = 100.0 

# --- 4. Boundary Conditions ---
# Let's hold the ends of the rod at 0 degrees (Dirichlet Boundary Condition)
# These values will not change in the loop.
u[:, 0] = 0.0 # 'Left' end
u[:, -1] = 0.0 # 'Right' end

# Calculate the stability constant 'r'
r = alpha * delta_t / delta_x**2

# Loop over time (skip the first row, which is t=0)
for n in range(0, nt):
    # Update the interior points (from index 1 to nx-2)
    # We use vectorization (slicing) instead of a second loop for speed
    u[n+1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2])
    
    # Note: We do not update u[:, 0] or u[:, -1] because those are boundaries.

# Create the x-axis for plotting
x = np.linspace(0, L, nx)

plt.figure(figsize=(10, 6))

# Plot initial condition
plt.plot(x, u[0, :], label='t=0 (Initial)')

# Plot t=100 (Step 100)
if nt > 100:
    plt.plot(x, u[100, :], label='t=100 steps')

# Plot final time
plt.plot(x, u[-1, :], label=f't={nt} steps (Final)')

plt.title('1D Heat Diffusion')
plt.xlabel('Position (x)')
plt.ylabel('Temperature (u)')
plt.legend()
plt.grid(True)
plt.show()