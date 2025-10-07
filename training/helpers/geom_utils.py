"""
Collection of functions for multi view geometry.
"""

import numpy as np
import cv2

def load_K_Rt_from_P(P: np.ndarray):
    """
    Inputs
      P: (3,4) projection matrix, P = K [R | t], world coords

    Returns
      intrinsics: (4,4) with K in the top-left, K[2,2] == 1
      pose:       (4,4) world->camera (view) matrix: [R | -R*C]
    """
    # Decompose
    out = cv2.decomposeProjectionMatrix(P)
    K, R, C_h = out[0], out[1], out[2]

    K = K / K[2, 2]                         # normalize so K[2,2] == 1
    C = (C_h[:3] / C_h[3]).reshape(3)       # camera center in world coords

    # Fix possible sign flips from OpenCV so fx, fy > 0
    if K[0, 0] < 0:
        K[:, 0] *= -1
        R[0, :] *= -1
    if K[1, 1] < 0:
        K[:, 1] *= -1
        R[1, :] *= -1

    # Build outputs
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[:3, :3] = K.astype(np.float32)

    pose = np.eye(4, dtype=np.float32)      # world -> camera
    pose[:3, :3] = R.astype(np.float32)
    pose[:3, 3]  = (-R @ C).astype(np.float32)

    return intrinsics, pose