import numpy as np
import matplotlib.pyplot as plt

# ========================================
# Domain and grid parameters
# ========================================

Nx, Ny = 150, 150  # Number of grid points in x and y
dx = 1.0  # Grid spacing in x
dt = 0.3  # Time step
T = 600 # Total number of time steps

V0 = 1.0  # Base growth velocity
alpha = 0.2  # Arrest rate
array_size = 12  # Number of domains 

# ========================================
# Level-set Fields
# ========================================
phi = np.ones((Nx, Ny))  # Level-set function (growth front)
A = np.ones((Nx, Ny))    # Activity / arrest field

# ========================================
# Domain generation in a grid pattern
# ========================================
num_rows = array_size
num_cols = array_size
spacing_x = Nx/ (array_size + 1) 
spacing_y = Ny/ (array_size + 1)

domains = []
np.random.seed(42)  # For reproducibility

side_direction = 'left'  # Side from which to generate seeds
seed_thickness = 30  # Thickness of the seed region

for r in range(num_rows):
    for c in range(num_cols):
        cx = int((c + 1) * spacing_x)
        cy = int((r + 1) * spacing_y)
        
        if side_direction == 'left':
            if cx > seed_thickness: continue
        elif side_direction == 'right':
            if cx < (Nx - seed_thickness): continue
        elif side_direction == 'top':
            if cy > seed_thickness: continue
        elif side_direction == 'bottom':
            if cy < (Ny - seed_thickness): continue
            
        # st = np.random.randint(0, 6) # Random start time   
        st = 0  # Synchronized start time
        domains.append((int(cx), int(cy), st))


# ========================================
# Functions
# ========================================

def grad_norm(f):
    grads = np.gradient (f, dx)
    fx = grads[0]
    fy = grads[1]
    return np.sqrt(fx**2 + fy**2) + 1e-8 # Avoid division by zero

def heaviside(x): 
    return (x > 0).astype(float) 

# ========================================
# Time evolution
# ========================================

snap_phi = []
snap_A   = []

for t in range(T):
    for (x, y, ts) in domains:
        if t == ts:
            phi[x, y] = -1.0
            
    V   = V0 * A
    phi = phi - dt * V * grad_norm(phi) # Level-set update
    A   = A - dt * alpha * A * heaviside(-phi) # Activity update
    
    if t % (T // 4) == 0:
        snap_phi.append(phi.copy())
        snap_A.append(A.copy())
        
# ========================================
# Visualization
# ========================================

fig, axes = plt.subplots(2, len(snap_phi), figsize=(20,8), constrained_layout=True)
im1 = None
im2 = None


for i in range(len(snap_phi)):
    im1 = axes[0,i].imshow(snap_phi[i], cmap="gist_earth", vmin=-5, vmax=1)
    axes[0,i].set_title(f"phi (t={i*(T//4)})", fontsize=16)
    axes[0,i].axis("off")

    im2 = axes[1,i].imshow(snap_A[i], cmap="viridis", vmin=0, vmax=1)
    axes[1,i].set_title(f"A (t={i*(T//4)})", fontsize=16)
    axes[1,i].axis("off")
    
# ==========================================
# Colorbars
# ==========================================

# Colorbar for Solidification Age / Depth (Phi)
cbar1 = fig.colorbar(im1, ax=axes[0, :], location='right', fraction=0.02)
cbar1.set_label("Solidification Age / Depth ($\phi$)\n[Ochre: Old Solid $\leftrightarrow$ Green: Recent Boundary]", fontsize=10)

# Colorbar for Activity (A)
cbar2 = fig.colorbar(im2, ax=axes[1, :], location='right', fraction=0.02)
cbar2.set_label("Growth Activity ($A$)\n[Yellow: Active $\leftrightarrow$ Purple: Frozen]", fontsize=10)


plt.suptitle(f"Dynamic Growth (Total domains: {len(domains)})")
plt.show()