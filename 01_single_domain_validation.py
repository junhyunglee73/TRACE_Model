# ------------------------------------------------
# Minimal simulation to validate front propagation
# and irreversible arrest for a single domain
# ------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

# ========================================
# Domain and grid parameters
# ========================================

Nx, Ny = 250, 250  # Number of grid points in x and y
dx = 1.0  # Grid spacing in x
dt = 0.3  # Time step
T = 840  # Total number of time steps

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
domains_x, domains_y = Nx//2, Ny//2  # Center of the domain
domains = [(domains_x, domains_y, 0)]

# ========================================
# Functions
# ========================================

def grad_norm(f):
    grads = np.gradient (f, dx)
    fx = grads[0]
    fy = grads[1]
    return np.sqrt(fx**2 + fy**2) + 1e-8 # Avoid division by zero

def heaviside(x): 
    # Returns 1 in transformed regions (phi < 0), enforcing irreversible arrest
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


for i in range(len(snap_phi)):
    im1 = axes[0,i].imshow(snap_phi[i], cmap="gist_earth", vmin=-5, vmax=1)
    axes[0,i].set_title(f"phi (t={i*(T//4)})")
    axes[0,i].axis("off")

    im2 = axes[1,i].imshow(snap_A[i], cmap="viridis", vmin=0, vmax=1)
    axes[1,i].set_title(f"A (t={i*(T//4)})")
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