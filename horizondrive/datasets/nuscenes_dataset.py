import csv
import os
import random
import logging
import numpy as np
from typing import Any, Dict, Literal, List, Optional
import torch
import torchvision.transforms as transforms
import pickle
import logging
try:
    from pyquaternion import Quaternion
except ImportError:
    Quaternion = None
from horizondrive.datasets.base_dataset import (
    BaseDataset,
    get_normalized_intrinsics
)
from horizondrive.utils.camera.pinhole import PinholeCamera
from horizondrive.utils.editing_utils import (
    nuscenes_build_tracks_and_minmax_bboxes,
)
from PIL import Image

import traceback

NUSCENES_HDMAP_LAYERS = [
    "road_segment",
    "lane",
    "lane_connector",
    "ped_crossing",
    "walkway",
    "stop_line",
    "road_divider",
    "lane_divider",
]

NUSCENES_HDMAP_COLORS = {
    "road_segment": (105, 105, 105),     # dark grey - road area (E2E: roadedges/roadside)
    "lane": (255, 255, 255),             # white - solid lane line (E2E: solid_lanes/Solid/White/Single)
    "lane_connector": (200, 200, 200),    # light grey - dotted lane connector (E2E: solid_lanes/Dotted/White/Single)
    "ped_crossing": (255, 0, 255),       # magenta - pedestrian crossing (E2E: crosswalks)
    "walkway": (211, 211, 211),          # light grey - walkway (E2E: roadedges/groundside)
    "stop_line": (255, 255, 0),          # yellow - stop line (E2E: stoplines/waiting_area)
    "road_divider": (255, 255, 0),       # yellow - road divider (E2E: solid_lanes/Solid/Yellow/Single)
    "lane_divider": (255, 255, 255),     # white - lane divider (E2E: solid_lanes/Solid/White/Single)
}

NUSCENES_CAMERA_NAME_MAP = {
    "camera_front": "CAM_FRONT",
    "camera_front_left": "CAM_FRONT_LEFT",
    "camera_front_right": "CAM_FRONT_RIGHT",
    "camera_rear": "CAM_BACK",
    "camera_rear_left": "CAM_BACK_LEFT",
    "camera_rear_right": "CAM_BACK_RIGHT",
}





class nuScenesDataset(BaseDataset):
    """
    nuScenes video dataset used by the open-source eval path.
    """

    def __init__(
        self,
        ann_path: str,
        mode="train",
        ref_camera=None,
        camera_names=["camera_front"],
        video_sample_stride: int = 1,
        video_length_drop_start: float = 0.0,
        video_length_drop_end: float = 1.0,
        i2v_random_mask_probs: Dict[Literal["first_image", "random_middle_image", "random_first_n_images", "drop_last"], float] = {"first_image": 1.0},
        valid_conditions: list = [],
        conditions_kwargs: Dict[str, Any] = {},
        bbox_use_ap = False,
        samples_path = None,
        map_dataroot = None,
        hdmap_layers = None,
        hdmap_patch_radius = 100.0,
        hdmap_line_width = 2,
        temporal_compression_ratio: int = 4,
        use_t_variant_noise: bool = True,
        video_length = 17,
        clip_start_stride: int = 1,
        start_on_firstframe: Optional[bool] = None,
        start_on_keyframe: Optional[bool] = None,
    ):

        self.samples_path = samples_path
        self.map_dataroot = map_dataroot or samples_path
        self.temporal_compression_ratio = temporal_compression_ratio
        self.use_t_variant_noise = use_t_variant_noise
        self.clip_start_stride = clip_start_stride
        self.hdmap_layers = hdmap_layers or NUSCENES_HDMAP_LAYERS
        self.hdmap_patch_radius = hdmap_patch_radius
        self.hdmap_line_width = hdmap_line_width
        self._nuscenes_maps = {}
        self.ann_path = ann_path
        self.ann_file = ann_path
        self.video_sample_stride = video_sample_stride
        self.micro_frame_size = None
        self.balance_keywords = None
        self.mode = mode
        self.video_length = video_length
        self.start_on_keyframe = bool(start_on_keyframe) if start_on_keyframe is not None else False
        self.start_on_firstframe = bool(start_on_firstframe) if start_on_firstframe is not None else (mode != 'train')
        
        self.data_infos = self.load_annotations(self.ann_file)
        
        if len(self.clip_infos) == 0:
            raise ValueError("No video data found in annotation file")
        self.ref_camera = ref_camera
        self.camera_names = camera_names
        self.nuscenes_camera_names = [
            NUSCENES_CAMERA_NAME_MAP.get(name, name) for name in self.camera_names
        ]
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        self.i2v_random_mask_probs = i2v_random_mask_probs
        self.length = len(self.clip_infos)
        
        self.valid_conditions = valid_conditions
        self.conditions_kwargs = conditions_kwargs
        self.text_embeds = None
        self.negative_text_embeds = None
        if os.path.exists("prompt_embeds_empty.pt"):
            self.text_embeds = torch.load("prompt_embeds_empty.pt", weights_only=True)["prompt_embeds"]
        if os.path.exists("prompt_embeds_negative.pt"):
            self.negative_text_embeds = torch.load("prompt_embeds_negative.pt", weights_only=True)["prompt_embeds"]
        self.use_valid_flag = True
        self.CLASSES = [
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
        ]

        self.clip_token_to_clip_idx = {
            self.data_infos[clip[0]]["token"]: clip_idx
            for clip_idx, clip in enumerate(self.clip_infos)
            if len(clip) > 0
        }

    def build_clips(self, data_infos, scene_tokens):
        """Since the order in self.data_infos may change on loading, we
        calculate the index for clips after loading.

        Args:
            data_infos (list of dict): loaded data_infos
            scene_tokens (2-dim list of str): 2-dim list for tokens to each
            scene

        Returns:
            2-dim list of int: int is the index in self.data_infos
        """
        self.token_data_dict = {
            item['token']: idx for idx, item in enumerate(data_infos)}
        if self.balance_keywords is not None:
            data_infos, scene_tokens = self.balance_annotations(
                data_infos, scene_tokens)
        all_clips = []
        clip_source_infos = []
        skip1, skip2 = 0, 0
        # clip_start_stride: controls spacing between adjacent clip starting
        # positions within a scene.  Default 1 (every frame) when
        # start_on_keyframe is True the effective stride is ~6 for 12Hz
        # interpolated NuScenes data.  Set e.g. 50 to sample one clip every
        # 50 frames, greatly reducing redundancy.
        clip_start_stride = int(self.clip_start_stride)
        clip_force_last_frame = bool(getattr(self, "clip_force_last_frame", True))
        clip_force_first_frame = bool(getattr(self, "clip_force_first_frame", True))
        for sid, scene in enumerate(scene_tokens):
            if self.video_length == "full":
                clip = [self.token_data_dict[token] for token in scene]
                if self.micro_frame_size is not None:
                    res = len(clip) % self.micro_frame_size - 1
                    if res > 0:
                        clip = clip[:-res]
                all_clips.append(clip)
                clip_source_infos.append({"scene_idx": sid, "start": 0})
            else:
                assert isinstance(self.video_length, int)
                if sid in []:
                    logging.info(f"Got {len(all_clips)} for sid={sid}.")
                if self.start_on_firstframe:
                    first_frames = [0]
                else:
                    first_frames = range(len(scene) - self.video_length + 1)
                last_start = len(scene) - self.video_length
                has_first_clip = False
                has_last_clip = False
                for start in first_frames:
                    if clip_start_stride > 1 and start % clip_start_stride != 0:
                        continue
                    if self.start_on_keyframe and ";" in scene[start]:
                        skip1 += 1
                        continue
                    if self.start_on_keyframe and len(scene[start]) >= 33:
                        skip2 += 1
                        continue
                    if start == 0:
                        has_first_clip = True
                    if start == last_start:
                        has_last_clip = True
                    clip = [self.token_data_dict[token]
                            for token in scene[start: start + self.video_length]]
                    all_clips.append(clip)
                    clip_source_infos.append({"scene_idx": sid, "start": start})
                # Ensure a clip starting at frame 0
                if clip_force_first_frame and not has_first_clip and len(scene) >= self.video_length:
                    clip = [self.token_data_dict[token]
                            for token in scene[0: self.video_length]]
                    all_clips.append(clip)
                    clip_source_infos.append({"scene_idx": sid, "start": 0})
                # Ensure a clip ending at the last frame of the scene
                if clip_force_last_frame and not has_last_clip and last_start >= 0:
                    clip = [self.token_data_dict[token]
                            for token in scene[last_start: last_start + self.video_length]]
                    all_clips.append(clip)
                    clip_source_infos.append({"scene_idx": sid, "start": last_start})
        logging.info(f"[{self.__class__.__name__}] Got {len(scene_tokens)} "
                     f"continuous scenes. Cut into {self.video_length}-clip, "
                     f"which has {len(all_clips)} in total. We skip {skip1} + "
                     f"{skip2} = {skip1 + skip2} possible starting frames. "
                     f"start_on_firstframe={self.start_on_firstframe}, "
                     f"clip_start_stride={clip_start_stride}")
        self.clip_source_infos = clip_source_infos
        return all_clips
    
    def load_annotations(self, ann_file: str) -> list:
        """
        Read the annotation file and return the ordered frame metadata.
        
        Args:
            ann_path (str): Path to the annotation file pickle.
        
        Returns:
            list: List of video data dictionaries
        """
        
        with open(ann_file, 'rb') as f:
            data = pickle.load(f)
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.video_sample_stride]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]
        self.scene_tokens = data["scene_tokens"]
        self.clip_infos = self.build_clips(data_infos, self.scene_tokens)

        return data_infos
    
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_bboxes_id = np.array(info["gt_box_ids"])[mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # Keep the raw box tensor; no center convention conversion is applied here.
        gt_bboxes_3d = torch.as_tensor(gt_bboxes_3d, dtype=torch.float32)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_bboxes_id=gt_bboxes_id,
        )
        return anns_results, mask

    def get_data_info_single(self, index: int) -> Dict[str, Any]:
        info = self.data_infos[index]
        data = dict(
            token=info["token"],
            sample_idx=info['token'],
            lidar_path=info["lidar_path"],
            sweeps=info["sweeps"],
            timestamp=info["timestamp"],
            location=info["location"],
        )
        add_key = [
            "description",
            "timeofday",
            "visibility",
            "flip_gt",
        ]
        for key in add_key:
            if key in info:
                data[key] = info[key]

        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        data["lidar2ego"] = lidar2ego

        data["image_paths"] = []
        data["camera_names"] = []
        data["lidar2camera"] = []
        data["lidar2image"] = []
        data["camera2ego"] = []
        data["camera_intrinsics"] = []
        data["camera2lidar"] = []
        data["ego2global"] = []

        for camera_name, camera_info in info["cams"].items():
            data["camera_names"].append(camera_name)
            data["image_paths"].append(camera_info["data_path"])

            lidar2camera_r = np.linalg.inv(camera_info["sensor2lidar_rotation"])
            lidar2camera_t = (
                camera_info["sensor2lidar_translation"] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t
            data["lidar2camera"].append(lidar2camera_rt.T)

            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_info["camera_intrinsics"]
            data["camera_intrinsics"].append(camera_intrinsics)

            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            data["lidar2image"].append(lidar2image)

            camera2ego = np.eye(4).astype(np.float32)
            camera2ego[:3, :3] = Quaternion(
                camera_info["sensor2ego_rotation"]
            ).rotation_matrix
            camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            data["camera2ego"].append(camera2ego)

            camera2lidar = np.eye(4).astype(np.float32)
            camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            data["camera2lidar"].append(camera2lidar)

            ego2global = np.eye(4).astype(np.float32)
            ego2global[:3, :3] = Quaternion(
                camera_info["ego2global_rotation"]
            ).rotation_matrix
            ego2global[:3, 3] = camera_info["ego2global_translation"]
            data["ego2global"].append(ego2global)

        annos, mask = self.get_ann_info(index)
        if "visibility" in data:
            data["visibility"] = data["visibility"][mask]
        data["ann_info"] = annos
        return data

    def load_clip(self, clip):
        frames = []
        for frame in clip:
            frame_info = self.get_data_info_single(frame)
            info = self.data_infos[frame]
            frames.append(frame_info)
        return frames

    def _resolve_clip_indices(self, index: int, requested_length: Optional[int] = None):
        clip = self.clip_infos[index]
        if requested_length is None:
            return clip

        requested_length = int(requested_length)

        if requested_length <= len(clip):
            if self.mode != 'train':
                start = 0
            else:
                max_start = len(clip) - requested_length
                start = random.randint(0, max_start) if max_start > 0 else 0
            return clip[start:start + requested_length]

        source = self.clip_source_infos[index]
        scene = self.scene_tokens[source["scene_idx"]]
        start = int(source["start"])
        # Clamp start so that start + requested_length <= scene_len
        max_start = len(scene) - requested_length
        if max_start < 0:
            raise ValueError(
                f"nuScenes clip idx={index} scene too short for infer_image_len={requested_length}: "
                f"scene_len={len(scene)}"
            )
        start = min(start, max_start)
        end = start + requested_length
        return [self.token_data_dict[token] for token in scene[start:end]]

    def get_data_info(self, index, infer_image_len: Optional[int] = None):
        """We should sample from clip_infos."""
        clip = self._resolve_clip_indices(index, requested_length=infer_image_len)
        frames = self.load_clip(clip)
        return frames

    def read_and_process_images(
        self,
        image_paths: List[str],
        target_height: int,
        target_width: int,
        image_transforms: Optional[transforms.Compose] = None,
    ) -> torch.Tensor:
        if image_transforms is None:
            image_transforms = transforms.Compose([
                transforms.Resize((target_height, target_width)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])

        processed_frames = []

        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((target_width, target_height))
                img_tensor = transforms.functional.to_tensor(img)
                processed_frames.append(img_tensor)
            except Exception as e:
                print(f"Warning: Failed to load image {img_path}: {e}")
                raise ValueError(f"Failed to load image: {img_path}")

        if not processed_frames:
            raise ValueError("No frames were loaded.")

        pixel_values = torch.stack(processed_frames)
        pixel_values = image_transforms(pixel_values)
        return pixel_values


    def _get_nuscenes_map(self, map_name):
        if self.map_dataroot is None:
            raise ValueError("map_dataroot/samples_path is required for nuScenes hdmap rendering")
        if map_name not in self._nuscenes_maps:
            # Patch matplotlib style before importing nuscenes devkit,
            # which calls plt.style.use('seaborn-v0_8-whitegrid') at import time.
            import matplotlib.pyplot as _plt
            _orig_use = _plt.style.use
            _plt.style.use = lambda *_: None
            try:
                from nuscenes.map_expansion.map_api import NuScenesMap
            finally:
                _plt.style.use = _orig_use
            self._nuscenes_maps[map_name] = NuScenesMap(
                dataroot=self.map_dataroot,
                map_name=map_name,
            )
        return self._nuscenes_maps[map_name]


    @staticmethod
    def _iter_hdmap_xy_sequences(geom):
        if geom.is_empty:
            return
        if geom.geom_type == "LineString":
            yield np.asarray(geom.coords, dtype=np.float32)[:, :2]
        elif geom.geom_type == "MultiLineString":
            for line in geom.geoms:
                yield np.asarray(line.coords, dtype=np.float32)[:, :2]
        elif geom.geom_type == "Polygon":
            yield np.asarray(geom.exterior.coords, dtype=np.float32)[:, :2]
        elif geom.geom_type == "MultiPolygon":
            for polygon in geom.geoms:
                yield np.asarray(polygon.exterior.coords, dtype=np.float32)[:, :2]
        elif hasattr(geom, "geoms"):
            for sub_geom in geom.geoms:
                yield from nuScenesDataset._iter_hdmap_xy_sequences(sub_geom)

    @staticmethod
    def _split_xy_by_distance(xy, center_xy, max_dist):
        if xy.shape[0] < 2:
            return
        dist = np.linalg.norm(xy - center_xy[None, :], axis=1)
        keep = dist <= max_dist
        start = None
        for i, valid in enumerate(keep):
            if valid and start is None:
                start = i
            elif not valid and start is not None:
                if i - start >= 2:
                    yield xy[start:i]
                start = None
        if start is not None and xy.shape[0] - start >= 2:
            yield xy[start:]

    def get_nuscenes_hdmap_records_for_clip(self, frames, camera_index):
        nusc_map = self._get_nuscenes_map(frames[0]["location"])
        ego_xy = np.asarray([frame["ego2global"][camera_index][:2, 3] for frame in frames])
        r = float(self.hdmap_patch_radius)
        box_coords = (
            float(ego_xy[:, 0].min() - r),
            float(ego_xy[:, 1].min() - r),
            float(ego_xy[:, 0].max() + r),
            float(ego_xy[:, 1].max() + r),
        )
        return nusc_map, nusc_map.get_records_in_patch(
            box_coords,
            self.hdmap_layers,
            mode="intersect",
        )

    def render_nuscenes_hdmap_frame(self, frame, camera_index, height, width, hdmap_context=None):
        try:
            from shapely.geometry import LineString
        except ImportError as exc:
            raise ImportError(
                "Rendering nuScenes hdmap requires nuscenes-devkit and shapely. "
                "Install them in the training environment or disable the hdmap condition."
            ) from exc

        if hdmap_context is None:
            nusc_map = self._get_nuscenes_map(frame["location"])
            ego_xy = frame["ego2global"][camera_index][:2, 3]
            r = float(self.hdmap_patch_radius)
            box_coords = (
                float(ego_xy[0] - r),
                float(ego_xy[1] - r),
                float(ego_xy[0] + r),
                float(ego_xy[1] + r),
            )
            records = nusc_map.get_records_in_patch(
                box_coords,
                self.hdmap_layers,
                mode="intersect",
            )
        else:
            nusc_map, records = hdmap_context
        intrinsics = frame["camera_intrinsics"][camera_index][:3, :3].astype(np.float32).copy()
        image_path = frame["image_paths"][camera_index]
        try:
            origin_w, origin_h = Image.open(image_path.replace('../data/nuscenes', self.samples_path)).size
        except Exception:
            origin_w, origin_h = width, height
        intrinsics[0, :] *= width / origin_w
        intrinsics[1, :] *= height / origin_h

        cam2ego = frame["camera2ego"][camera_index]
        ego2global = frame["ego2global"][camera_index]
        cam2global = ego2global @ cam2ego
        ego_xy = ego2global[:2, 3]
        max_dist = float(self.hdmap_patch_radius)

        camera_model = PinholeCamera(
            fx=intrinsics[0, 0],
            fy=intrinsics[1, 1],
            cx=intrinsics[0, 2],
            cy=intrinsics[1, 2],
            w=width,
            h=height,
            device="cpu",
        )

        # Collect polylines by layer for per-layer colored rendering
        polylines_by_layer = {}
        for layer_name, tokens in records.items():
            color = NUSCENES_HDMAP_COLORS.get(layer_name, (255, 255, 255))
            is_polygon = layer_name in nusc_map.non_geometric_polygon_layers
            layer_polylines = []
            for token in tokens:
                record = nusc_map.get(layer_name, token)
                if layer_name == "drivable_area":
                    geoms = [
                        nusc_map.extract_polygon(poly_token)
                        for poly_token in record["polygon_tokens"]
                    ]
                elif is_polygon:
                    geoms = [nusc_map.extract_polygon(record["polygon_token"])]
                elif "line_token" in record:
                    geoms = [nusc_map.extract_line(record["line_token"])]
                elif "node_tokens" in record:
                    coords = [
                        (
                            nusc_map.get("node", node_token)["x"],
                            nusc_map.get("node", node_token)["y"],
                        )
                        for node_token in record["node_tokens"]
                    ]
                    geoms = [LineString(coords)] if len(coords) >= 2 else []
                else:
                    geoms = []

                for geom in geoms:
                    for xy in self._iter_hdmap_xy_sequences(geom):
                        for xy_segment in self._split_xy_by_distance(xy, ego_xy, max_dist):
                            # Convert 2D global xy to 3D global xyz (z=0 ground plane)
                            pts_3d = np.concatenate(
                                [xy_segment, np.zeros((xy_segment.shape[0], 1), dtype=np.float32)],
                                axis=1,
                            )
                            layer_polylines.append(pts_3d)

            if layer_polylines:
                polylines_by_layer[layer_name] = (layer_polylines, color)

        # Render each layer with depth-adaptive line thickness via PinholeCamera.draw_line_depth
        per_layer_projections = []
        for layer_name, (layer_polylines, color) in polylines_by_layer.items():
            proj = camera_model.draw_line_depth(
                cam2global,
                layer_polylines,
                radius=int(self.hdmap_line_width),
                colors=np.array(color),
                segment_interval=0.25,
            )
            # draw_line_depth may return (1, H, W, 3) or (H, W, 3); squeeze batch dim
            if proj.ndim == 4:
                proj = proj[0]
            per_layer_projections.append(proj)

        if per_layer_projections:
            hdmap = np.maximum.reduce(per_layer_projections)
        else:
            hdmap = np.zeros((height, width, 3), dtype=np.uint8)

        return hdmap

    def get_batch(
        self,
        idx: int,
        resoultion: list = [17, 240, 480],
        conditions: list = [],
        infer_image_len = None,
        valid_mode = None,
    ):
        """
        Load and process a video batch

        Args:
            idx (int): Index of the video to load

        Returns:
            tuple: (pixel_values, text, data_type) where pixel_values is torch.Tensor [F, C, H, W]
        """

        video_sample_n_frames, train_height, train_width = resoultion
        if infer_image_len is not None:
            if infer_image_len > 0:
                video_sample_n_frames = int(infer_image_len)
            else:
                # infer_image_len <= 0: dynamic length, use full clip
                clip = self.clip_infos[idx % self.length]
                t_cr = getattr(self, 'temporal_compression_ratio', 4)
                video_sample_n_frames = len(clip) - (len(clip) - 1) % t_cr

        frames = self.get_data_info(idx, infer_image_len=video_sample_n_frames)
        assert video_sample_n_frames == len(frames), f"mismatch video length between {video_sample_n_frames} and {len(frames)}"

        camera2ego = frames[0]['camera2ego'][0]
        lidar2ego = frames[0]['lidar2ego']
        camera2lidar = [frame['camera2lidar'][0] for frame in frames]
        ego2global = [frame['ego2global'][0] for frame in frames]

        camera_intrinsics = frames[0]['camera_intrinsics'][0]

        old_prefix = '../data/nuscenes'
        all_cam_images_paths = []
        data_camera_name = self.nuscenes_camera_names[0]
        for frame in frames:
            for path in frame['image_paths']:
                if data_camera_name in path:
                    new_path = path.replace(old_prefix, self.samples_path)
                    all_cam_images_paths.append(new_path)
                    break
        
        tokens = [frame['token'] for frame in frames]
        clip_id = tokens[0]

        bbox_pixel_values = None
        hdmap_pixel_values = None
        bbox_ap_conds = None

        pixel_values = self.read_and_process_images(
            all_cam_images_paths,
            train_height,
            train_width,
        )
        if "hdmap" in conditions:
            hdmap_frames = []
            camera_index = frames[0]["camera_names"].index(data_camera_name)
            hdmap_context = self.get_nuscenes_hdmap_records_for_clip(frames, camera_index)
            for frame in frames:
                hdmap_frame = self.render_nuscenes_hdmap_frame(
                    frame,
                    camera_index,
                    train_height,
                    train_width,
                    hdmap_context=hdmap_context,
                )
                hdmap_frames.append(hdmap_frame)
            hdmap_pixel_values = np.asarray(hdmap_frames)
            hdmap_pixel_values = torch.from_numpy(hdmap_pixel_values).permute(0, 3, 1, 2).contiguous()
            hdmap_pixel_values = hdmap_pixel_values / 255.0
            hdmap_pixel_values = hdmap_pixel_values * 2.0 - 1.0
        delta_T = torch.eye(4, device=pixel_values.device, dtype=pixel_values.dtype).unsqueeze(0).repeat(pixel_values.shape[0], 1, 1)
        delta_T_origin = delta_T.clone()

        if "action" in conditions:
            vcs2worlds = ego2global
            init_world2vcs = np.linalg.inv(vcs2worlds[0])
            F = video_sample_n_frames
            vcs2world_list = []
            actions = []
            delta_T = []

            for i in range(F):
                vcs2world = init_world2vcs @ vcs2worlds[i]
                vcs2world_list.append(vcs2world)
 
            for i in range(F - 1):
                T_i = vcs2world_list[i]
                T_next = vcs2world_list[i + 1]
                T_rel = np.linalg.inv(T_i) @ T_next
                delta = T_rel[:3, 3]
                # Compute yaw change (same as E2EUnifiedDataset)
                R_rel = T_rel[:3, :3]
                delta_yaw = np.arctan2(R_rel[1, 0], R_rel[0, 0])
                delta_yaw_theta = delta_yaw * 180.0 / np.pi
                action = np.array([delta[0], delta[1], delta_yaw_theta], dtype=np.float32)
                actions.append(torch.tensor(action, dtype=torch.float32))
                delta_T.append(torch.tensor(T_rel, dtype=torch.float32))

            actions = torch.stack(actions, dim=0)
            last_action = actions[-1:] 
            actions = torch.cat([actions, last_action], dim=0)[:, :3]

            delta_T = torch.stack(delta_T, dim=0)
            last_delta_T = delta_T[-1:]
            delta_T = torch.cat([delta_T, last_delta_T], dim=0)

            delta_T_origin = delta_T.clone()

        if "bbox" in conditions:
            masked_target = []
            ann_infos = [frame['ann_info'] for frame in frames]

            bbox_ap_conds = nuscenes_build_tracks_and_minmax_bboxes(
                ann_infos,
                pixel_values,
                masked_target,
                fill_value=np.nan,
                clip_id=clip_id,
                valid_mode=valid_mode,
                conditions=conditions,
                camera_names=self.nuscenes_camera_names,
                delta_T=delta_T,
                delta_T_origin=delta_T_origin,
                camera_intrinsics=camera_intrinsics,
                cam2lidar=camera2lidar,
                cam2vcs=camera2ego,
                lidar2vcs=lidar2ego,
            )

            bbox_pixel_values = bbox_ap_conds["bbox_render"] * 2.0 - 1.0

        text = ''
 
        sample = {
            "pixel_values": pixel_values,
            "text": text,
            "clip_id": clip_id,
            "data_type": 'video',
            "idx": idx,
            "conditions": {},
            "source_idx": [],
            "has_hood": False,
        }
        sample["encoder_hidden_states"] = self.text_embeds
        sample["negative_encoder_hidden_states"] = self.negative_text_embeds
        # Always fill addition_conditions with the same sub-key structure as E2EUnifiedDataset
        n_cam = len(self.camera_names)
        n_frames = pixel_values.shape[0] // n_cam
        sample["addition_conditions"] = {
            "worlds2vcs": torch.zeros(n_cam * n_frames, 4, 4),
            "cam_intrics": torch.zeros(n_cam, 3, 3),
            "vcs2cam_list": torch.zeros(n_cam, 4, 4),
        }
        if bbox_pixel_values is not None:
            sample["conditions"]["bbox"] = bbox_pixel_values
        if hdmap_pixel_values is not None:
            sample["conditions"]["hdmap"] = hdmap_pixel_values
        if "action" in conditions:
            sample["conditions"]["action"] = actions
        return sample
    
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(
        self,
        kwargs,
    ) -> Dict[str, Any]:
        
        idx = kwargs["idx"]
        model_mode = kwargs["model_mode"]
        resolution = kwargs["resolution"]
        conditions = kwargs.get("conditions", [])
        num_condition_images = kwargs.get("num_condition_images", None)
        infer_image_len = kwargs.get("infer_image_len", None)
        validation_mode = kwargs.get("validation_mode", None)

        if model_mode != "i2v" and num_condition_images is not None:
            raise ValueError(f"num_condition_images is not supported for {model_mode} model")

        assert model_mode in ["t2v", "i2v"], f"Invalid mode: {model_mode}"

        while True:
            sample = {}
            try:
                sample = self.get_batch(
                    idx,
                    resolution,
                    conditions,
                    infer_image_len=infer_image_len,
                    valid_mode=validation_mode,
                )
                if len(sample) > 0:
                    break
            
            except Exception as e:
                if validation_mode is None:
                    traceback.print_exc()
                    print(f"Error processing video: {e}")
                    idx = random.randint(0, self.length - 1)
                else:
                    traceback.print_exc()
                    return None

        if model_mode == "i2v":
            mask = self.get_mask(sample["pixel_values"].size(), "first_image", num_condition_images)
            sample["mask"] = mask
        sample["model_mode"] = model_mode
        return sample
