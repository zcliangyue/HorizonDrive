# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
from tqdm import tqdm

from horizondrive.utils.minimap_utils import cuboid3d_to_polyline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

CLASS_COLORS = {
    "Car":  [255, 0, 0],
    "Truck": [0, 0, 255],
    "Pedestrian": [0, 255, 0],
    "Cyclist": [255, 255, 0],
    "Others": [255, 255, 255],
}


def create_bbox_projection(all_object_info, camera_poses, valid_frame_ids, camera_model):
    """
    Create a projection of bounding boxes on the minimap.
    Args:
        all_object_info: dict, containing all object info
        camera_poses: np.ndarray, shape (N, 4, 4), dtype=np.float32, camera to world transformation matrix
        camera_model: CameraModel, camera model
        valid_frame_ids: list[int], valid frame ids
        draw_heading: bool, whether to draw heading on the bounding boxes
        diff_color: bool, whether to use different colors for dynamic and static objects

    Returns:
        np.ndarray, shape (N, H, W, 3), dtype=np.uint8, projected bounding boxes on canvas
    """
    bbox_projections = []

    for i in valid_frame_ids:
        current_object_info = all_object_info[f"{i:06d}.all_object_info.json"]

        polylines_cars = []
        polylines_trucks = []
        polylines_pedestrians = []
        polylines_cyclists = []
        polylines_others = []

        # sort tracking ids. avoid jittering when drawing bbox.
        tracking_ids = list(current_object_info.keys())
        tracking_ids.sort()

        for tracking_id in tracking_ids:
            object_info = current_object_info[tracking_id]

            if object_info['object_type'] not in CLASS_COLORS:
                if object_info['object_type'] == "Bus":
                    object_info['object_type'] = "Truck"
                elif object_info['object_type'] == 'Vehicle':
                    object_info['object_type'] = "Car"
                else:
                    object_info['object_type'] = "Others"

            object_to_world = np.array(object_info['object_to_world'])
            object_lwh = np.array(object_info['object_lwh'])
            cuboid_eight_vertices = build_cuboid_bounding_box(object_lwh[0], object_lwh[1], object_lwh[2], object_to_world)
            polyline = cuboid3d_to_polyline(cuboid_eight_vertices)

            # draw by the object type
            if object_info['object_type'] == "Car":
                polylines_cars.append(polyline)
            elif object_info['object_type'] == "Truck":
                polylines_trucks.append(polyline)
            elif object_info['object_type'] == "Pedestrian":
                polylines_pedestrians.append(polyline)
            elif object_info['object_type'] == "Cyclist":
                polylines_cyclists.append(polyline)
            else:
                polylines_others.append(polyline)

        cars_bbox_projection = camera_model.draw_line_depth(camera_poses[i], polylines_cars, radius=5, colors=np.array(CLASS_COLORS["Car"]))
        trucks_bbox_projection = camera_model.draw_line_depth(camera_poses[i], polylines_trucks, radius=5, colors=np.array(CLASS_COLORS["Truck"]))
        pedestrians_bbox_projection = camera_model.draw_line_depth(camera_poses[i], polylines_pedestrians, radius=5, colors=np.array(CLASS_COLORS["Pedestrian"]))
        cyclists_bbox_projection = camera_model.draw_line_depth(camera_poses[i], polylines_cyclists, radius=5, colors=np.array(CLASS_COLORS["Cyclist"]))
        others_bbox_projection = camera_model.draw_line_depth(camera_poses[i], polylines_others, radius=5, colors=np.array(CLASS_COLORS["Others"]))

        # combine the dynamic and static bbox projection
        bbox_projection = np.maximum.reduce([cars_bbox_projection, trucks_bbox_projection, pedestrians_bbox_projection, cyclists_bbox_projection, others_bbox_projection])
        bbox_projections.append(bbox_projection)

    return np.concatenate(bbox_projections, axis=0)


def interpolate_pose(prev_pose, next_pose, t):
    """
    new pose = (1 - t) * prev_pose + t * next_pose.
    - linear interpolation for translation
    - slerp interpolation for rotation

    Args:
        prev_pose: np.ndarray, shape (4, 4), dtype=np.float32, previous pose
        next_pose: np.ndarray, shape (4, 4), dtype=np.float32, next pose
        t: float, interpolation factor

    Returns:
        np.ndarray, shape (4, 4), dtype=np.float32, interpolated pose

    Note:
        if input is list, also return list.
    """
    input_is_list = isinstance(prev_pose, list)
    prev_pose = np.array(prev_pose)
    next_pose = np.array(next_pose)

    prev_translation = prev_pose[:3, 3]
    next_translation = next_pose[:3, 3]
    translation = (1 - t) * prev_translation + t * next_translation

    prev_rotation = R.from_matrix(prev_pose[:3, :3])
    next_rotation = R.from_matrix(next_pose[:3, :3])
    
    times = [0, 1]
    rotations = R.from_quat([prev_rotation.as_quat(), next_rotation.as_quat()])
    rotation = Slerp(times, rotations)(t)

    new_pose = np.eye(4)
    new_pose[:3, :3] = rotation.as_matrix()
    new_pose[:3, 3] = translation

    if input_is_list:
        return new_pose.tolist()
    else:
        return new_pose
    

def interpolate_bbox(all_object_info, valid_frame_ids):
    """
    Interpolate bbox from 10Hz to 30Hz.
    Args:
        all_object_info: dict, containing all object info. Keys will be 
            {frame_id.06d}.all_object_info.json, where frame_id has a interval of 3.
            For example, 000000.all_object_info.json, 000003.all_object_info.json, 000006.all_object_info.json, etc.

            For one key, the value is a dict, containing all object info for that frame.
            "000000.all_object_info.json": {
                "1": {
                    "object_to_world": 4x4 matrix,
                    "object_lwh": 3-length array,
                    "object_is_moving": bool,
                    "object_type": str,
                },
                "2": {
                    ...
                },  
            }

            Here "1" is the tracking id, and the value is the object info for that frame.

        valid_frame_ids: list[int], valid frame ids

    Returns:
        dict, containing interpolated object info
    """
    interpolated_all_object_info = {}

    for frame_id in valid_frame_ids:
        # no need to interpolate
        if f"{frame_id:06d}.all_object_info.json" in all_object_info:
            interpolated_all_object_info[f"{frame_id:06d}.all_object_info.json"] = \
                all_object_info[f"{frame_id:06d}.all_object_info.json"]
        else:
            # find the nearest frame with object info
            prev_frame_id = frame_id
            next_frame_id = frame_id

            while f"{prev_frame_id:06d}.all_object_info.json" not in all_object_info and prev_frame_id >= 0:
                prev_frame_id -= 1
            while f"{next_frame_id:06d}.all_object_info.json" not in all_object_info and next_frame_id <= max(valid_frame_ids):
                next_frame_id += 1

            # usually prev_frame_id can be found. If next_frame_id is out of range, we just duplicate prev_frame_id
            if next_frame_id > max(valid_frame_ids):
                interpolated_all_object_info[f"{frame_id:06d}.all_object_info.json"] = \
                    interpolated_all_object_info[f"{prev_frame_id:06d}.all_object_info.json"]
                continue

            # interpolate the object info from the previous and next frame
            prev_object_info = all_object_info[f"{prev_frame_id:06d}.all_object_info.json"]
            next_object_info = all_object_info[f"{next_frame_id:06d}.all_object_info.json"]
            
            # tracking ids in the previous and next frame
            prev_tracking_ids = set(prev_object_info.keys())
            next_tracking_ids = set(next_object_info.keys())

            # common tracking ids in the previous and next frame
            common_tracking_ids = prev_tracking_ids & next_tracking_ids

            t = (frame_id - prev_frame_id) / (next_frame_id - prev_frame_id)

            interpolated_object_info = {}
            # interpolate the object info from the previous and next frame
            for tracking_id in common_tracking_ids:
                prev_pose = np.array(prev_object_info[tracking_id]['object_to_world'])
                next_pose = np.array(next_object_info[tracking_id]['object_to_world'])
                interpolated_pose = interpolate_pose(prev_pose, next_pose, t)
                interpolated_object_info[tracking_id] = {}
                interpolated_object_info[tracking_id]['object_to_world'] = interpolated_pose.tolist()

                prev_lwh = np.array(prev_object_info[tracking_id]['object_lwh'])
                next_lwh = np.array(next_object_info[tracking_id]['object_lwh'])
                interpolated_lwh = (1 - t) * prev_lwh + t * next_lwh
                interpolated_object_info[tracking_id]['object_lwh'] = interpolated_lwh.tolist()

                interpolated_object_info[tracking_id]['object_is_moving'] = prev_object_info[tracking_id]['object_is_moving']
                interpolated_object_info[tracking_id]['object_type'] = prev_object_info[tracking_id]['object_type']

            interpolated_all_object_info[f"{frame_id:06d}.all_object_info.json"] = interpolated_object_info

    return interpolated_all_object_info


def quaternion_mean(quaternions):
    """
    Compute the mean of quaternions (resolving double-cover issue).

    Args:
        quaternions (np.ndarray): Array of quaternions, shape (N, 4).

    Returns:
        np.ndarray: Mean quaternion, shape (4,).
    """

    quaternions = np.array(quaternions)

    # Unify quaternion signs (ensure w is positive)
    for i in range(len(quaternions)):
        if quaternions[i, 0] < 0:  # Flip quaternion if w is negative
            quaternions[i] = -quaternions[i]

    # Calculate mean quaternion
    mean_quaternion = np.mean(quaternions, axis=0)
    mean_quaternion /= np.linalg.norm(mean_quaternion)  # Normalize

    return mean_quaternion

def rotation_matrix_mean(rotation_matrices):
    """
    Compute the mean rotation matrix from a set of rotation matrices (based on quaternions).

    Args:
        rotation_matrices (list of np.ndarray): List of rotation matrices (3x3).

    Returns:
        np.ndarray: Mean rotation matrix (3x3).
    """

    # Convert rotation matrices to Rotation objects
    rotations = [R.from_matrix(R_matrix) for R_matrix in rotation_matrices]

    # Convert Rotation objects to quaternions
    quaternions = [rotation.as_quat() for rotation in rotations]

    # Compute mean quaternion
    mean_quaternion = quaternion_mean(quaternions)

    # Convert mean quaternion back to Rotation object
    mean_rotation = R.from_quat(mean_quaternion)

    # Convert Rotation object back to rotation matrix
    mean_rotation_matrix = mean_rotation.as_matrix()

    return mean_rotation_matrix


def build_cuboid_bounding_box(dimXMeters, dimYMeters, dimZMeters, cuboid_transform=np.eye(4)):
    """
    Args
        dimXMeters, dimYMeters, dimZMeters: float, the dimensions of the cuboid
        cuboid_transform: 4x4 numpy array, the transformation matrix from the cuboid coordinate to the other coordinate

        z
        ^
        |   y
        | / 
        |/
        o----------> x  (heading)

           3 ---------------- 0
          /|                 /|
         / |                / |
        2 ---------------- 1  |
        |  |               |  |
        |  7 ------------- |- 4
        | /                | /
        6 ---------------- 5 
        
    Returns
        8x3 numpy array: the 8 vertices of the cuboid
    """
    # Build the cuboid bounding box
    cuboid = np.array([
        [dimXMeters / 2, dimYMeters / 2, dimZMeters / 2],
        [dimXMeters / 2, -dimYMeters / 2, dimZMeters / 2],
        [-dimXMeters / 2, -dimYMeters / 2, dimZMeters / 2],
        [-dimXMeters / 2, dimYMeters / 2, dimZMeters / 2],
        [dimXMeters / 2, dimYMeters / 2, -dimZMeters / 2],
        [dimXMeters / 2, -dimYMeters / 2, -dimZMeters / 2],
        [-dimXMeters / 2, -dimYMeters / 2, -dimZMeters / 2],
        [-dimXMeters / 2, dimYMeters / 2, -dimZMeters / 2]
    ])
    cuboid = np.hstack([cuboid, np.ones((8, 1))]) # [8, 4]
    cuboid = np.dot(cuboid_transform, cuboid.T).T 
    return cuboid[:, :3]


def object_tfm_to_heading(tfm):
    """
    Args:
        tfm: 4x4 numpy array, the transformation matrix
    Returns:
        heading_vector: [3,] numpy array, the heading of the object
    """
    if isinstance(tfm, list):
        tfm = np.array(tfm)
        
    heading_vector = tfm[:3, 0]
    heading_vector = heading_vector / np.linalg.norm(heading_vector)
    return heading_vector
