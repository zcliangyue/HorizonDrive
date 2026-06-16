import einops
import torch
import random
from typing import Any, Dict, Literal
from abc import abstractmethod
from torch.utils.data.dataset import Dataset
import traceback

def get_normalized_intrinsics(K, width, height):
    K_norm = torch.zeros_like(K)
    K_norm[0, 0] = K[0, 0] / width
    K_norm[1, 1] = K[1, 1] / height
    K_norm[0, 2] = K[0, 2] / width - 0.5
    K_norm[1, 2] = K[1, 2] / height - 0.5
    K_norm[2, 2] = 1.0
    K_norm_homo = torch.eye(4, dtype=K_norm.dtype)
    K_norm_homo[:3, :3] = K_norm
    return K_norm_homo


class BaseDataset(Dataset):
    """
    A dataset class for video-only training.
    
    This dataset simplifies ImageVideoDataset by only processing videos
    and includes random video segment selection for long videos.
    
    Features:
    1. Only processes video data (removes image processing logic)
    2. Random video segment selection for long videos
    3. Direct pixel value return like original ImageVideoDataset
    
    Args:
        ann_path (str): Path to annotation file (YAML)
    """
    
    def __init__(
        self,
        ann_path: str,
        mode="train",
        video_repeat=0,
        ref_camera=None,
        camera_names=["camera_front"],
        video_sample_stride: int = 1,
        video_length_drop_start: float = 0.0,
        video_length_drop_end: float = 1.0,
        text_drop_ratio: float = 0.1,
        i2v_random_mask_probs: Dict[Literal["first_image", "random_middle_image", "random_first_n_images", "drop_last", "first_11_images"], float] = {"first_image": 1.0},  # random mask type for inpainting
        valid_conditions: list = [],
        conditions_kwargs: Dict[str, Any] = {},
        bbox_use_ap = False,
        samples_path = None, # for nuScenes dataset
    ):
        super().__init__()
        
        self.dataset = self.read_ann_path(ann_path)
        if video_repeat > 0:
            self.dataset = self.dataset * video_repeat
        
        if len(self.dataset) == 0:
            raise ValueError("No video data found in annotation file")
        
        # self.debug_hdmap_dir = debug_hdmap_dir
        self.mode = mode
        self.ref_camera = ref_camera
        self.camera_names = camera_names
        self.video_sample_stride = video_sample_stride
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end
        self.text_drop_ratio = text_drop_ratio
        self.i2v_random_mask_probs = i2v_random_mask_probs
        self.length = len(self.dataset)
        
        self.valid_conditions = valid_conditions
        self.conditions_kwargs = conditions_kwargs

    @abstractmethod   
    def get_batch(
        self,
        idx: int,
        resoultion: list = [17, 240, 480],
    ):
        """
        Get a batch of video data.
        
        Args:
            idx (int): Index of the video sample
            resoultion (list): Resolution of the video frames
            
        Returns:
            dict: Contains pixel_values, text, data_type, and idx
        """
        raise NotImplementedError("Subclasses should implement this method")
        

    def get_mask(self, pixel_values_shape, mask_type, num_condition_images=None):
        """
        Generate mask for inpainting according to ref_camera and ref_image
        Args:
            frame_num (int): Number of frames in the video
        Returns:
            torch.Tensor: Mask tensor of shape [frame_num * len(self.camera_names), 1, H, W]
        """
        nf, c, h, w = pixel_values_shape
        n_cam = len(self.camera_names)
        n_t = nf // n_cam
        mask = torch.ones((n_cam, n_t, 1, h, w), dtype=torch.uint8)   # 1 means masked; 0 keeps the frame as a condition input.

        # keep all frames of ref_camera
        if self.ref_camera is not None:
            ref_cam_idx = self.camera_names.index(self.ref_camera)
            mask[ref_cam_idx, :] = 0

        s = getattr(self, "temporal_compression_ratio", 1)
        if num_condition_images is not None:
            # fixed mode
            assert (num_condition_images - 1) % s == 0, f"num_condition_images is not true"
            mask[:, :num_condition_images] = 0
        elif mask_type == "first_image":
            mask[:, 0] = 0
        elif mask_type == "first_half_image":
            mask[:, : n_t // 2] = 0
        elif mask_type == "random_middle_image":
            n_latent = (n_t - 1) // s + 1
            random_latent_idx = random.randint(1, n_latent - 1) - 1
            mask[:, random_latent_idx*s+1:(random_latent_idx+1)*s+1] = 0
        elif mask_type == "random_first_n_images":
            n_latent = (n_t - 1) // s + 1
            random_latent_num = random.randint(1, n_latent - 1) - 1
            mask[:, :random_latent_num*s+1] = 0
        elif mask_type == "drop_last":
            mask[:, :-s] = 0
        elif mask_type == "first_11_images":
            mask[:, :11] = 0
        else:
            raise ValueError(f"Invalid mask type: {mask_type}")
        
        mask = einops.rearrange(mask, "n_cam n_t c h w -> (n_cam n_t) c h w")
        return mask
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(
        self,
        kwargs,
    ) -> Dict[str, Any]:
        """
        Get a video sample.
        
        Returns:
            dict: Contains pixel_values, text, data_type, and idx (same format as ImageVideoDataset)
        """
        idx = kwargs["idx"]
        model_mode = kwargs["model_mode"]
        resolution = kwargs["resolution"]
        conditions = kwargs.get("conditions", [])
        # print('e2e: resolution', resolution, 'conditions', conditions)
        num_condition_images = kwargs.get("num_condition_images", None)
        validation_mode = kwargs.get("validation_mode", None)

        if model_mode != "i2v" and num_condition_images is not None:
            raise ValueError(f"num_condition_images is not supported for {model_mode} model")

        assert model_mode in ["t2v", "i2v"], f"Invalid mode: {model_mode}"
        while True:
            sample = {}
            try:
                sample = self.get_batch(idx, resolution, conditions, valid_mode=validation_mode)
                if len(sample) > 0:
                    break
            
            except Exception as e:
                if validation_mode is None:
                    print(f"Error processing video: {e}")
                    idx = random.randint(0, self.length - 1)
                else:
                    # traceback.print_exc()
                    print(f"Error processing video: {idx} {e}")
                    return None
        
        # Add inpainting functionality if enabled (same as ImageVideoDataset)

        if model_mode == "i2v":
            mask_type = random.choices(list(self.i2v_random_mask_probs.keys()), weights=list(self.i2v_random_mask_probs.values()), k=1)[0]
            mask = self.get_mask(sample["pixel_values"].size(), mask_type, num_condition_images)
            mask_pixel_values = sample["pixel_values"] * (1 - mask)
            sample["mask_pixel_values"] = mask_pixel_values
            sample["mask"] = mask
            
            clip_pixel_values = sample["pixel_values"][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            sample["clip_pixel_values"] = clip_pixel_values
            
            ref_pixel_values = sample["pixel_values"][0].unsqueeze(0)
            if (mask == 1).all():
                ref_pixel_values = torch.ones_like(ref_pixel_values) * -1
            sample["ref_pixel_values"] = ref_pixel_values
        sample["model_mode"] = model_mode
        return sample
