import numpy as np
import matplotlib.pyplot as plt

# ========================================
# Domain and grid parameters
# ========================================

Nx, Ny = 250, 250  # Number of grid points in x and y
dx = 0.8  # Grid spacing in x
dt = 0.3  # Time step
T = 500  # Total number of time steps

V0 = 1.0  # Base growth velocity
alpha = 0.2  # Arrest rate


# ========================================
# Level-set Fields
# ========================================
phi = np.ones((Nx, Ny))  # Level-set function (growth front)
A = np.ones((Nx, Ny))    # Activity / arrest field

# ========================================
# Domain generation in a grid pattern
# ========================================

num_domains = 50  # Number of domains in each row and column
np.random.seed(42)  # For reproducibility

center_x, center_y = Nx//2, Ny//2

max_radius = min(Nx, Ny) // 2 - 5  # radius of the outer circle
hole_ratio = 0.9                   # ratio for the inner hole
min_radius = max_radius * hole_ratio

domains = []

while len(domains) < num_domains:
    rx = np.random.randint(0, Nx)
    ry = np.random.randint(0, Ny)
    
    dist = np.sqrt((rx - center_x)**2 + (ry - center_y)**2)
    
    if min_radius < dist < max_radius:
        rt = np.random.randint(0, 20)  # Random start time
        # rt = 0  # Synchronized start time
        domains.append((rx, ry, rt))


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

# Create circular mask
Y, X = np.ogrid[:Nx, :Ny]
dist_from_center = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
circular_mask = dist_from_center > max_radius # Mask for points outside the outer circle


for i in range(len(snap_phi)):
    curr_phi = snap_phi[i].copy()
    curr_A = snap_A[i].copy()
    curr_phi[circular_mask] = np.nan
    curr_A[circular_mask] = np.nan

    im1 = axes[0,i].imshow(curr_phi, cmap="gist_earth", vmin=-5, vmax=1)
    axes[0,i].set_title(f"phi (t={i*(T//4)})", fontsize=16)
    axes[0,i].axis("off")

    im2 = axes[1,i].imshow(curr_A, cmap="viridis", vmin=0, vmax=1)
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