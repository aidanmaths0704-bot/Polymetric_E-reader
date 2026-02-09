import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # <--- This line forces the Linux window to open!
import matplotlib.pyplot as plt

# ==========================================
# 1. PARAMETERS & CONFIGURATION
# ==========================================
N = 64            # Grid size (64x64 pixels)
h = 1.0          
dt = 0.01         # Time step
steps = 50000     # Max steps (The code will likely stop earlier!)
plot_interval = 1500  # How often to save an image for the graph

# Physical Parameters
M = 1.0     #Mobility Coefficient High M flows like water seperation is easy and vice versa        
gamma = 2.0       
a = 1.0     #Temperature Sensitivity (The Accelerator) (How strongly the Temperature affects the system)
            #If a is large, a small drop in temperature below Tc creates a massive driving force for separation and vice versa)     
b = 1.0     #Saturation Strength (The Brakes that keep the system from going too far)      
            #Since Phi**4 grows extremely fast, as Phi increases, b is responsible for stabilizing the system.  
            #(cannot pack infinite polymer into one spot)
Tc = 0.5    #Critical Temperature for Demixing (The temperature at which the system will start to separate into two phases)
            #Like how the critical temp of water is 0C, the critical temp of the polymer is 0.5 dimensionless units.
            #If T > Tc, the system will be in the homogeneous phase.
            #If T < Tc, the system will be in the demixed phase.
            #If T = Tc, the system will be in the critical point.
            #Temperature in this model is dimensionless (unitless) T here is a ratio or a scaled difference
            #relative to the critical point Tc, not absolute temperature.

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def get_laplacian_2D(field, h):
    """ Calculates curvature using 5-point stencil with Neumann boundaries """
    padded = np.pad(field, pad_width=1, mode='edge') #mode = 'edge' for zero flux boundaries
    top    = padded[0:-2, 1:-1]
    bottom = padded[2:,   1:-1]
    left   = padded[1:-1, 0:-2]
    right  = padded[1:-1, 2:]
    center = padded[1:-1, 1:-1]
    return (top + bottom + left + right - 4 * center) / (h**2) #from 2D diffusion equation (P&G module)

def get_free_energy_derivative(phi, T, Tc, a, b):
    return a * (T - Tc) * phi + b * phi**3 #This was given in the CW prompt

# ==========================================
# 3. INITIALIZATION
# ==========================================
# Deep Quench Temperature (Set to -0.5 to ensure full +/- 1 separation)
T_profile = np.ones((N, N)) * -0.5  

# Random Initial Noise
np.random.seed(42)
phi = np.random.uniform(-0.1, 0.1, (N, N))

# ==========================================
# 4. SMART SIMULATION LOOP
# ==========================================
print("Starting Smart Simulation...")
history = []
history.append(phi.copy()) # Save initial state

# Variables for tracking stability
saturation_counter = 0     
saturation_threshold = 50   # Steps to wait after hitting +/- 1 before stopping
target_val = 1.0            # We want to hit 1.0 and -1.0
tolerance = 0.02            # Allow small error (e.g. 0.98 is "close enough")

# Variables for "Quiet Mode" printing
last_print_max = 0.0
last_print_min = 0.0
print_threshold = 0.05      # Only print if value changes by this much

for t in range(1, steps): # Start at 1 since we saved step 0
    
    # --- A. The Physics (Cahn-Hilliard) ---
    df_dphi = get_free_energy_derivative(phi, T_profile, Tc, a, b) #little push so the ball doesnt sit at top of the hill
    lap_phi = get_laplacian_2D(phi, h) # Measure how different a pixel is from its neighbors. This term drives the smoothing process.
    w = gamma * lap_phi - df_dphi #The net thermodynamic force for at every point. Tells the polymer: "Considering the local concentration gradient, how much should I move?"
    dphi_dt = -M * get_laplacian_2D(w, h)
    phi = phi + dphi_dt * dt
    
    # --- B. Get Current Stats ---
    curr_max = np.max(phi)
    curr_min = np.min(phi)
    
    # --- C. Smart Printing (Only print if things are changing) ---
    # We check if the max OR min has changed significantly since the last print
    change_in_max = abs(curr_max - last_print_max)
    change_in_min = abs(curr_min - last_print_min)
    
    # Always print on plot_intervals, otherwise only print if interesting
    if t % plot_interval == 0 or (change_in_max > print_threshold) or (change_in_min > print_threshold):
        print(f"Step {t}: Max={curr_max:.4f}, Min={curr_min:.4f}")
        last_print_max = curr_max
        last_print_min = curr_min
        
        # Save snapshot if it's a plot interval
        if t % plot_interval == 0:
            history.append(phi.copy())

    # --- D. Early Stopping Check ---
    # Check if we have hit the targets (accounting for overshoot/undershoot)
    is_black_saturated = (curr_max >= target_val - tolerance)
    is_white_saturated = (curr_min <= -target_val + tolerance)
    
    if is_black_saturated and is_white_saturated:
        saturation_counter += 1
    else:
        saturation_counter = 0 # Reset counter if we dip back
        
    if saturation_counter >= saturation_threshold:
        print(f"\n✅ SATURATION REACHED at Step {t}!")
        print(f"Final Max: {curr_max:.4f}, Final Min: {curr_min:.4f}")
        print("Stopping early to save time.")
        
        # Save the final state if we haven't just saved it
        if t % plot_interval != 0:
            history.append(phi.copy())
        break

# ==========================================
# 5. VISUALIZATION
# ==========================================
num_plots = len(history)
print(f"Displaying {num_plots} snapshots.")

# Adjust figure size based on how many plots we have
fig, axes = plt.subplots(1, num_plots, figsize=(3 * num_plots, 4))

# Handle case if there's only 1 plot (rare, but prevents crash)
if num_plots == 1: axes = [axes]

for i, ax in enumerate(axes):
    current_data = history[i]
    
    # We use vmin=-1.1 and vmax=1.1 to handle the slight overshoot visual
    im = ax.imshow(current_data, cmap='Greys', origin='lower', vmin=-1.1, vmax=1.1)
    
    # Label the steps correctly
    # If it's the last one, label it "Final"
    if i == num_plots - 1:
        step_label = "Final State"
    elif i == 0:
        step_label = "Initial (Step 0)"
    else:
        # Approximate step number based on interval
        step_label = f"Step ~{i * plot_interval}"
        
    ax.set_title(step_label)
    ax.axis('off')

plt.suptitle(f"E-Reader Simulation (Stopped at Saturation)", fontsize=16)
plt.colorbar(im, ax=axes.ravel().tolist() if num_plots > 1 else axes, label=r"Concentration ($\phi$)")
plt.show()