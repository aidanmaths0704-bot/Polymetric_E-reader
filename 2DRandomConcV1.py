import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & CONFIGURATION
# ==========================================
N = 64            # Grid size (64x64 pixels)
dx = 1.0          
dt = 0.01         # Time step
steps = 300000     # Total simulation steps (Run longer for better results!)
plot_interval = 40000

# Physical Parameters
M = 1.0           
sigma = 2.0       
a = 1.0          
b = 1.0           
Tc = 0.5          

# ==========================================
# 2. HELPER FUNCTIONS (2D Version)
# ==========================================
def get_laplacian_2D(field, dx):
    """
    Calculates 2D Laplacian using a 5-point stencil (Up, Down, Left, Right).
    Uses np.pad to handle Neumann Boundaries (Zero Flux) automatically.
    """
    # Pad the field with its own edge values (Neumann condition)
    padded = np.pad(field, pad_width=1, mode='edge')
    
    # Neighbors (from the padded array)
    top    = padded[0:-2, 1:-1]
    bottom = padded[2:,   1:-1]
    left   = padded[1:-1, 0:-2]
    right  = padded[1:-1, 2:]
    center = padded[1:-1, 1:-1] # This is just the original field
    
    # Finite Difference Formula: (Sum of neighbors - 4*Center) / dx^2
    laplacian = (top + bottom + left + right - 4 * center) / (dx**2)
    
    return laplacian

def get_free_energy_derivative(phi, T, Tc, a, b):
    return a * (T - Tc) * phi + b * phi**3

# ==========================================
# 3. INITIALIZATION
# ==========================================
# Initialize 2D Temperature Grid
# Start with cold everywhere
T_profile = np.ones((N, N)) * 0.4  

# OPTIONAL: Add a "Hot Spot" (Eraser) in the middle
# This simulates a heated region where the screen should turn grey
# T_profile[20:44, 20:44] = 0.6  # Uncomment this line to see the effect!

# Initialize Phi (2D Random Noise)
np.random.seed(42)
phi = np.random.uniform(-0.1, 0.1, (N, N))

# ==========================================
# 4. MAIN SIMULATION LOOP
# ==========================================
print("Starting 2D simulation...")
history = []

for t in range(steps):
    
    # 1. Chemical Potential
    df_dphi = get_free_energy_derivative(phi, T_profile, Tc, a, b)
    lap_phi = get_laplacian_2D(phi, dx)
    w = sigma * lap_phi - df_dphi
    
    # 2. Diffusion
    lap_w = get_laplacian_2D(w, dx)
    dphi_dt = -M * lap_w
    
    # 3. Update
    phi = phi + dphi_dt * dt
    
    # Save snapshots
    if t % plot_interval == 0:
        history.append(phi.copy())

# Save final state
history.append(phi.copy())
print("Simulation complete.")

# ==========================================
# 5. VISUALIZATION (Auto-Scaling)
# ==========================================
print(f"Final Data Range: Min={np.min(history[-1]):.4f}, Max={np.max(history[-1]):.4f}")

num_plots = len(history)
fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))

for i, ax in enumerate(axes):
    # REMOVE vmin=-1, vmax=1
    # Let Python auto-scale the colors (vmin=None)
    current_data = history[i]
    im = ax.imshow(current_data, cmap='Greys', origin='lower')
    
    ax.set_title(f"Step {i * plot_interval}")
    ax.axis('off')

plt.suptitle(f"2D Simulation (Auto-Scaled Colors)", fontsize=16)
plt.tight_layout()
plt.show()