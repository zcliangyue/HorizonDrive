# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np

def cuboid3d_to_polyline(cuboid3d_eight_vertices):
    """
    Convert cuboid3d to polyline

    Args:
        cuboid3d_eight_vertices: np.ndarray, shape (8, 3), dtype=np.float32, 
            eight vertices of the cuboid3d
    
    Returns:
        polyline: np.ndarray, shape (N, 3), dtype=np.float32, 
            polyline vertices
    """
    if isinstance(cuboid3d_eight_vertices, list):
        cuboid3d_eight_vertices = np.array(cuboid3d_eight_vertices)

    connected_vertices_indices = [0,1,2,3,0,4,5,6,7,4,5,1,2,6,7,3]
    connected_polyline = np.array(cuboid3d_eight_vertices)[connected_vertices_indices]

    return connected_polyline