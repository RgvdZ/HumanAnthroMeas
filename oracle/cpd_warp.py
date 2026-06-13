import numpy as np
import cv2
from pycpd import DeformableRegistration

def extract_boundary_points(mask, num_points=300):
    """Extract evenly spaced points from the silhouette boundary."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour found in mask")
    cnt = max(contours, key=cv2.contourArea).squeeze(1)  # (N, 2)
    if len(cnt) < num_points:
        # repeat points if not enough
        cnt = np.tile(cnt, (num_points // len(cnt) + 1, 1))[:num_points]
    else:
        # subsample evenly along contour
        indices = np.linspace(0, len(cnt)-1, num_points, dtype=int)
        cnt = cnt[indices]
    return cnt.astype(np.float64)   # (num_points, 2)

def compute_warp_cpd(mask_A, mask_B, num_points=300, max_iterations=100, tolerance=1e-5,
                     lambd=3.0, beta=2.0):
    """
    Compute dense warp field (in pixel units) that morphs mask_A into mask_B using CPD.
    Returns: warp_field (H, W, 2) with delta x, delta y.
    """
    H, W = mask_A.shape
    pts_A = extract_boundary_points(mask_A, num_points)   # (N,2)
    pts_B = extract_boundary_points(mask_B, num_points)   # (N,2)

    # Perform non-rigid CPD
    reg = DeformableRegistration(X=pts_B, Y=pts_A, max_iterations=max_iterations,
                                 tolerance=tolerance, alpha=lambd, beta=beta)
    reg.register()
    # reg.TY is the transformed points (pts_A -> approx pts_B)
    # Get the warp field for every pixel using the G (RBF) matrix from reg
    # The deformation is defined by G (kernel of source points) and W (weights)
    # Y_aligned = G * W + Y (approximately)
    # We can evaluate the warp at all pixel coordinates.

    # Create grid of pixel coordinates (H, W, 2)
    yv, xv = np.mgrid[0:H, 0:W]
    grid = np.stack([xv, yv], axis=-1).astype(np.float64).reshape(-1, 2)  # (H*W, 2)

    # Evaluate deformation at grid points using the CPD model's RBF
    # The transformation: f(x) = x + sum(w_i * G(x, y_i))
    # where G = exp(-||x - y||^2 / (2*beta^2))
    Y = reg.Y   # original source points (pts_A)
    W = reg.W   # weight matrix (num_points, 2)
    beta = reg.beta

    # Compute pairwise distances between grid and source points Y
    # grid (P,2) - Y (N,2) -> (P,N)
    diff = grid[:, None, :] - Y[None, :, :]  # (P, N, 2)
    sq_dist = np.sum(diff ** 2, axis=-1)      # (P, N)
    G = np.exp(-sq_dist / (2 * beta ** 2))    # (P, N)

    # Apply deformation: displacement = G @ W, output = grid + displacement
    displacement = G @ W  # (P, 2)
    # The CPD formulation adds the original Y displacement; ensure correct
    transformed_grid = grid + displacement

    warp_field = transformed_grid - grid      # delta_x, delta_y
    warp_field = warp_field.reshape(H, W, 2).astype(np.float32)

    # Mask out background (outside silhouette A)
    warp_field *= mask_A[:, :, None]
    return warp_field
