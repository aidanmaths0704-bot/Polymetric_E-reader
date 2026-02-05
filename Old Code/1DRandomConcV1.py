import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & CONFIGURATION
# ==========================================
N = 100           # Number of grid points
dx = 1.0          # Distance between points
L = N * dx        # Total length
dt = 0.001        # Time step (Keep small for stability)
steps = 20000     # Total simulation steps

# Physical Parameters
M = 1.0           # Mobility
sigma = 2.0       # Surface tension parameter
a = 1.0           # Free energy parameter
b = 1.0           # Free energy parameter
Tc = 0.5          # Critical Temperature

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_laplacian(field, dx):
    """ Calculates the 2nd derivative (curvature) using Finite Differences """
    laplacian = np.zeros_like(field)
    
    # Internal points
    laplacian[1:-1] = (field[0:-2] - 2*field[1:-1] + field[2:]) / dx**2
    
    # Boundary points (Neumann: no flux at edges)
    laplacian[0] = (field[1] - 2*field[0] + field[1]) / dx**2
    laplacian[-1] = (field[-2] - 2*field[-1] + field[-2]) / dx**2
    
    return laplacian

def get_free_energy_derivative(phi, T, Tc, a, b):
    """ derivative of f = a(T-Tc)phi^2/2 + b*phi^4/4 """
    return a * (T - Tc) * phi + b * phi**3

# ==========================================
# 3. INITIALIZATION
# ==========================================
x_axis = np.linspace(0, L, N)

# Constant Temperature Profile (T=0.4, which is less than Tc=0.5)
T_profile = np.ones(N) * 0.4  

# Random initial noise (Concentration phi)
np.random.seed(42) 
phi = np.random.uniform(-0.1, 0.1, N)

# ==========================================
# 4. MAIN LOOP
# ==========================================
history = [] 
plot_interval = 1000

print("Starting simulation...")

for t in range(steps):
    
    # 1. Chemical Potential components
    df_dphi = get_free_energy_derivative(phi, T_profile, Tc, a, b)
    lap_phi = get_laplacian(phi, dx)
    
    # 2. Calculate Chemical Potential (w)
    # w = sigma * lap_phi - df_dphi
    w = sigma * lap_phi - df_dphi
    
    # 3. Diffusion (Movement driven by chemical potential)
    # dphi/dt = -M * lap_w
    lap_w = get_laplacian(w, dx)
    dphi_dt = -M * lap_w
    
    # 4. Update
    phi = phi + dphi_dt * dt
    
    if t % plot_interval == 0:
        history.append(phi.copy())

print("Simulation complete.")

# ==========================================
# 5. VISUALIZATION
# ==========================================
plt.figure(figsize=(10, 6))

# Plot initial state (Grey dashed line)
plt.plot(x_axis, history[0], label='Initial (Mixed)', linestyle='--', color='grey')

# Plot intermediate state (Blue line)
mid_idx = len(history) // 2
plt.plot(x_axis, history[mid_idx], label=f'Step {mid_idx * plot_interval}', color='blue')

# Plot final state (Black thick line)
plt.plot(x_axis, history[-1], label='Final (Demixed)', color='black', linewidth=2)

plt.title(f"Polymeric E-Reader Simulation (T={T_profile[0]} < Tc={Tc})")
plt.xlabel("Position (x)")
# FIX: Added 'r' before the string to make it a raw string
plt.ylabel(r"Concentration ($\phi$)") 
plt.legend()
plt.grid(True)
plt.show()


# ==========================================
# 6. BARCODE VISUALIZATION (The "Top-Down" View)
# ==========================================
plt.figure(figsize=(12, 4))

# We take the final state (history[-1]) and stretch it vertically
# np.tile repeats the array to give it some 'height' so we can see it
barcode_height = 50 
barcode_data = np.tile(history[-1], (barcode_height, 1))

# Plot as an image
# cmap='Greys' makes Low values (-1) White and High values (+1) Black.
# vmin=-0.1 and vmax=0.1 sets the contrast range. 
# (Once your simulation runs longer and reaches +/- 1.0, change these to -1 and 1)
plt.imshow(barcode_data, cmap='Greys', aspect='auto', vmin=-0.1, vmax=0.1)

# Remove the Y-axis ticks because the height is fake/just for looks
plt.yticks([]) 
plt.xlabel("Position (x)")
plt.title(f"E-Reader Surface View (Step {steps})")
plt.colorbar(label="Concentration ($\phi$)", orientation='horizontal')

plt.tight_layout()
plt.show()