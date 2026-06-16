import csv
import json
import os
import random
import time
from typing import Optional
from copy import deepcopy
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from func_timeout import FunctionTimedOut, func_timeout
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from collections import defaultdict
from videox_fun.utils.utils import import_cls
from videox_fun.data.dataset_image_video import (
    VIDEO_READER_TIMEOUT,
    MultiVideoReader_contextmanager,
    VideoReader_contextmanager,
    get_video_reader_batch,
    resize_frame,
)
from horizondrive.datasets.base_dataset import (
    BaseDataset,
    get_normalized_intrinsics
)
from horizondrive.utils.camera.pinhole import PinholeCamera
import cv2
from PIL import Image
import numpy as np
from horizondrive.utils.bbox_utils import build_cuboid_bounding_box, cuboid3d_to_polyline, CLASS_COLORS

NUSC_NAME_TO_CLASS = {
    "car": "Car",
    "vehicle": "Car",
    "truck": "Truck",
    "trailer": "Truck",
    "bus": "Truck",
    "construction_vehicle": "Truck",
    "pedestrian": "Pedestrian",
    "bicycle": "Cyclist",
    "motorcycle": "Cyclist",
    "traffic_cone": "Others",
    "barrier": "Others",
}




def valid_bbox_mask(bboxes, min_size=16, max_aspect_ratio=6.0, edit=False):
    # bboxes: (T, 4) -> [y1, y2, x1, x2]
    not_nan = ~np.isnan(bboxes).any(axis=1)

    if edit:
        return not_nan

    heights = bboxes[:, 1] - bboxes[:, 0]
    widths  = bboxes[:, 3] - bboxes[:, 2]

    size_ok = (heights >= min_size) & (widths >= min_size)

    min_hw = np.minimum(heights, widths)
    min_hw = np.where(min_hw <= 0, 1e-6, min_hw)

    aspect = np.maximum(heights, widths) / min_hw
    aspect_ok = aspect <= max_aspect_ratio

    valid = not_nan & size_ok & aspect_ok

    return valid


def nuscenes_select_top_objects(frames, EXCLUDE_LABELS, top_k=10):
    """
    Select the globally highest-ranked object ids from per-frame object order.
    If fewer than top_k objects remain after filtering EXCLUDE_LABELS, keep all.

    Args:
        frames: list of per-frame object dictionaries.
        EXCLUDE_LABELS: categories to exclude.
        top_k: number of object ids to select.

    Returns:
        selected_ids: set[int]
    """
    rank_sum = defaultdict(int)
    count = defaultdict(int)

    for objs in frames:
        obj_gt_bboxes_id = objs['gt_bboxes_id']
        obj_gt_labels_3d = objs['gt_labels_3d']
        for rank in range(len(obj_gt_bboxes_id)):
            if obj_gt_labels_3d[rank] in EXCLUDE_LABELS:
                continue
            oid = obj_gt_bboxes_id[rank]
            rank_sum[oid] += rank
            count[oid] += 1

    if not count:
        return set(), [[] for _ in frames]

    avg_rank = {oid: rank_sum[oid] / count[oid] for oid in count}

    if len(avg_rank) <= top_k:
        selected_ids = set(avg_rank.keys())
    else:
        selected_ids = set(sorted(avg_rank, key=avg_rank.get)[:top_k])

    return selected_ids

def nuscenes_build_tracks_and_minmax_bboxes(
    obj_infos,
    pixel_values,
    EXCLUDE_LABELS,
    k_top=1000,
    fill_value=np.nan,
    clip_id=None,
    valid_mode=None,
    conditions=[],
    camera_names=[],
    delta_T=None,
    delta_T_origin=None,
    camera_intrinsics=None,
    cam2lidar=None,
    cam2vcs=None,
    lidar2vcs=None,
    valid_action_mode='default',
    valid_bbox_mode='default',
    valid_id_mode='default'):
    """
    Build object tracks and projected 2D boxes from full-video object metadata.

    Returns:
        - id_list: [k]
        - xyz_tracks: (k, T, 3)
        - bbox_tracks: (k, T, 4) -> (minh, maxh, minw, maxw)
    Missing or invalid projections are filled with fill_value.
    """

    def compute_transform_matrix(xyz, yaw):
        rotate_mat = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotate_mat
        transform_matrix[:3, 3] = xyz
        return transform_matrix


    T, C, H_new, W_new = pixel_values.shape

    # Select top objects by average rank while excluding ignored labels.
    selected_ids = nuscenes_select_top_objects(obj_infos, EXCLUDE_LABELS, k_top)
    id_list = sorted(list(selected_ids))
    k_eff = len(id_list)

    id2idx = {oid: idx for idx, oid in enumerate(id_list)}

    xyz_tracks = np.full((k_top, T, 3), fill_value, dtype=np.float32)
    minmax_tracks = np.full((k_top, T, 4), fill_value, dtype=np.float32)
    center_2D_tracks = np.full((k_top, T, 2), fill_value, dtype=np.float32)
    depth_tracks = np.full((k_top, T), fill_value, dtype=np.float32)

    bbox_projections_tensor = torch.zeros(
        (T, 3, H_new, W_new),
        dtype=pixel_values.dtype,
        device=pixel_values.device,
    )

    K = camera_intrinsics[:3, :3].copy()
    H_src, W_src = 900, 1600
    K[0, :] *= W_new / W_src
    K[1, :] *= H_new / H_src
    current_drift = torch.eye(4, device=delta_T.device)
    drift_list = [current_drift]
    for t in range(T-1):
        current_drift = torch.linalg.inv(delta_T_origin[t]) @ drift_list[t] @ delta_T[t]
        drift_list.append(current_drift.clone())
    accumulated_drifts = torch.stack(drift_list).cpu().numpy()
    vcs2cam = np.linalg.inv(cam2vcs)
    vcs2lidar = cam2lidar @ vcs2cam
    cam2lidar = vcs2lidar @ accumulated_drifts @ cam2vcs

    camera_model = PinholeCamera(
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        w=W_new,
        h=H_new,
        device="cpu",
    )

    obj_black_list = []
    obj_white_list = []
    id_white_list = []

    for t in range(T):
        objs = obj_infos[t]
        obj_gt_bboxes_3d = objs['gt_bboxes_3d']
        obj_gt_bboxes_id = objs['gt_bboxes_id']
        obj_gt_names = objs.get('gt_names', ['Car'] * len(obj_gt_bboxes_id))
        polylines_by_class = {k: [] for k in CLASS_COLORS}
        for i_obj in range(len(obj_gt_bboxes_id)):
            obj = obj_gt_bboxes_3d[i_obj]
            oid = obj_gt_bboxes_id[i_obj]

            xyz = np.array([
                obj[0],
                obj[1],
                obj[2]
            ])
            yaw = - (obj[6])

            object_lwh = np.array([
                obj[3],
                obj[4],
                obj[5]
            ])

            # nuScenes coordinates: +x points right, +y points forward.
            if valid_action_mode == "vx0":
                if t == 0 and abs(xyz[1]) < 0.7 and xyz[0] < 0.0:
                    obj_black_list.append(oid)
            if valid_action_mode == "moveleft":
                if t == 0 and xyz[0] < 1.0 and xyz[1] < 0.0:
                    obj_black_list.append(oid)
            if valid_action_mode == "moveright":
                if t == 0 and xyz[0] > -1.0 and xyz[1] < 0.0:
                    obj_black_list.append(oid)
            if valid_bbox_mode == 'vx1':
                if t==0 and abs(xyz[0]) < 1.0 and xyz[1] > 0.0 and xyz[1] < 10.0:
                    print('car in front of me!')
                    obj_white_list.append(oid)
                if oid in obj_white_list:
                    xyz[1] = xyz[1] + t * 0.25
            if valid_bbox_mode == 'moveright':
                if t==0 and xyz[0] < -1.0 and xyz[1] > 0.0 and xyz[1] < 20.0:
                    print('left car cut in!')
                    obj_white_list.append(oid)
                if oid in obj_white_list:
                    phase = (np.pi / 2) * t / T
                    xyz[1] = xyz[1] + t * (-0.6)
                    xyz[0] = xyz[0] + 3.5 * np.sin(phase)
            if valid_bbox_mode == 'moveleft':
                if t==0 and xyz[0] < 5.0 and xyz[0] > 1.0 and xyz[1] > 0.0 and xyz[1] < 20.0:
                    print('right car cut in!')
                    obj_white_list.append(oid)
                if oid in obj_white_list:
                    phase = (np.pi / 2) * t / T
                    xyz[1] = xyz[1] + t * (-0.2)
                    xyz[0] = xyz[0] - 3.5 * np.sin(phase)
            if valid_id_mode != 'default':
                if abs(xyz[0]) < 1.0 and xyz[1] > 0.0:
                    if t == 0:
                        id_white_list.append(oid)

            object_to_world = compute_transform_matrix(xyz, yaw)

            cuboid_eight_vertices = build_cuboid_bounding_box(object_lwh[0], object_lwh[1], object_lwh[2], object_to_world)  # [8, 3]

            polyline = cuboid3d_to_polyline(cuboid_eight_vertices)

            if not oid in obj_black_list:
                class_key = NUSC_NAME_TO_CLASS.get(obj_gt_names[i_obj].lower(), "Others")
                polylines_by_class[class_key].append(polyline)
            if (valid_id_mode != 'default') and (oid not in id_white_list):
                continue
            
            # minmax_2D requires all projected corners to lie inside the image.
            # center_2D only requires the object center to be visible.
            minmax_2D, center_2D, depth = camera_model.get_2D_minmax(cam2lidar[t], cuboid_eight_vertices, clamp_to_border=False)
            if not np.isnan(minmax_2D).any():
                minmax_2D = np.round(minmax_2D).astype(np.float32)
            if not np.isnan(center_2D).any():
                center_2D = np.round(center_2D).astype(np.float32)
            
            j = id2idx[oid]
            xyz_tracks[j, t] = xyz
            minmax_tracks[j, t] = minmax_2D
            center_2D_tracks[j, t] = center_2D
            depth_tracks[j, t] = depth

        per_class_projections = []
        for class_key, class_polylines in polylines_by_class.items():
            if not class_polylines:
                continue
            proj = camera_model.draw_line_depth(
                cam2lidar[t],
                class_polylines,
                radius=5,
                colors=np.array(CLASS_COLORS[class_key]),
            )
            per_class_projections.append(proj[0])
        if per_class_projections:
            bbox_projection = np.maximum.reduce(per_class_projections)
        else:
            bbox_projection = np.zeros((H_new, W_new, 3), dtype=np.uint8)

        bp_np = bbox_projection
        if bp_np.dtype == np.uint8:
            bp_np = bp_np.astype(np.float32) / 255.0
        else:
            bp_np = bp_np.astype(np.float32)

        bp_t = torch.from_numpy(bp_np).permute(2, 0, 1)

        bp_t = bp_t.to(device=pixel_values.device, dtype=pixel_values.dtype)

        bbox_projections_tensor[t] = bp_t.clamp_(0, 1)

    xyz_tracks = torch.from_numpy(xyz_tracks).to(device=pixel_values.device, dtype=torch.float32)
    minmax_tracks = torch.from_numpy(minmax_tracks).to(device=pixel_values.device, dtype=torch.float32)
    center_2D_tracks = torch.from_numpy(center_2D_tracks).to(device=pixel_values.device, dtype=torch.float32)

    return_item = {
        "xyz": xyz_tracks,                    # [k, T, 3]
        "minmax_2D": minmax_tracks,           # [k, T, 4]
        "center_2D": center_2D_tracks,        # [k, T, 2]
        "bbox_render": bbox_projections_tensor,   #[T, 3, H_new, W_new]
    }

    return return_item


