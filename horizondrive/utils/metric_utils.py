import os
import math
import torch
import numpy as np
from scipy.linalg import sqrtm
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import Tensor
from einops import rearrange
import torch.distributed as dist
from torchmetrics.image.fid import NoTrainInceptionV3, _compute_fid


def resolve_existing_path(path: Optional[str]) -> Optional[str]:
    return path if path and os.path.exists(path) else None


class StyleGanFVDMetric:
    pretrained_model_path="models/fvd/styleganv/i3d_torchscript.pt"
    def __init__(self, device='cpu'):
        self.device = device
        self.i3d = self.load_i3d_pretrained()
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1
        self.num_features = 400
        self.reset()

    def load_i3d_pretrained(self):
        i3D_WEIGHTS_URL = "https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt"
        if not os.path.exists(self.pretrained_model_path):
            print(f"preparing for download {i3D_WEIGHTS_URL}, you can download it by yourself.")
            os.system(f"wget {i3D_WEIGHTS_URL} -O {self.pretrained_model_path}")
        i3d = torch.jit.load(self.pretrained_model_path).eval().to(self.device).float()
        return i3d

    def preprocess_single(self, video, resolution=224, sequence_length=None):
        # video: CTHW, [0, 1]
        video = video.float()
        c, t, h, w = video.shape

        # temporal crop
        if sequence_length is not None:
            assert sequence_length <= t
            video = video[:, :sequence_length]

        # scale shorter side to resolution
        scale = resolution / min(h, w)
        if h < w:
            target_size = (resolution, math.ceil(w * scale))
        else:
            target_size = (math.ceil(h * scale), resolution)
        video = F.interpolate(video, size=target_size, mode='bilinear', align_corners=False)

        # center crop
        c, t, h, w = video.shape
        w_start = (w - resolution) // 2
        h_start = (h - resolution) // 2
        video = video[:, :, h_start:h_start + resolution, w_start:w_start + resolution]

        # [0, 1] -> [-1, 1]
        video = (video - 0.5) * 2
        return video.contiguous()

    @torch.no_grad()
    def get_fvd_feats(self, videos):
        detector_kwargs = dict(rescale=False, resize=False, return_features=True) # Return raw features before the softmax layer.
        feats = np.empty((0, 400)).astype(np.float32)  # Initialize an empty array to store features.
        x = torch.stack([self.preprocess_single(video) for video in videos]).to(self.device)
        feats = np.vstack([
            feats,
            self.i3d(x=x, **detector_kwargs).detach().float().cpu().numpy()
        ])
        return feats

    def frechet_distance(self, feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
        def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            mu = feats.mean(axis=0) # [d]
            sigma = np.cov(feats, rowvar=False) # [d, d]
            return mu, sigma
        mu_gen, sigma_gen = compute_stats(feats_fake)
        mu_real, sigma_real = compute_stats(feats_real)
        m = np.square(mu_gen - mu_real).sum()
        if feats_fake.shape[0]>1:
            s, _ = sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
            fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        else:
            fid = np.real(m)
        return float(fid)

    def _as_bvtchw(self, videos: Tensor, num_views: int) -> Tensor:
        """Convert eval video tensors to (B, V, T, C, H, W)."""
        if videos.dim() == 4:
            videos = videos.unsqueeze(0)
        if videos.dim() != 5:
            raise ValueError(f"Expected videos with shape (T,C,H,W) or (B,T,C,H,W), got {tuple(videos.shape)}")
        if videos.shape[1] % num_views != 0:
            raise ValueError(
                f"Flattened frame dimension {videos.shape[1]} is not divisible by num_views={num_views}"
            )
        return rearrange(videos, "b (v t) c h w -> b v t c h w", v=num_views)

    def _to_segments(
        self,
        videos: Tensor,
        video_length: int = 16,
        segment_stride: int = 16,
        num_views: int = 1,
    ) -> Tensor:
        """Convert eval video tensors into 16-frame FVD samples.

        The input layout is (T,C,H,W) or (B,T,C,H,W), where T may be flattened as
        (num_views * frames_per_view). The output layout is (N,C,16,H,W), with
        each non-overlapping 16-frame window treated as one FVD sample.
        """
        if video_length <= 0 or segment_stride <= 0:
            raise ValueError("video_length and segment_stride must be positive")

        videos = self._as_bvtchw(videos.detach(), num_views)
        frames_per_view = videos.shape[2]
        if frames_per_view < video_length:
            raise ValueError(
                f"Need at least {video_length} frames per view to compute FVD, got {frames_per_view}"
            )

        starts = list(range(0, frames_per_view - video_length + 1, segment_stride))
        segments = []
        for view_idx in range(num_views):
            for start_idx in starts:
                end_idx = start_idx + video_length
                segments.append(videos[:, view_idx, start_idx:end_idx].permute(0, 2, 1, 3, 4))
        return torch.cat(segments, dim=0)

    def update(
        self,
        videos: Tensor,
        real: bool,
        video_length: int = 16,
        segment_stride: int = 16,
        num_views: int = 1,
    ) -> int:
        """Accumulate FVD statistics from 16-frame segments."""
        segments = self._to_segments(
            videos,
            video_length=video_length,
            segment_stride=segment_stride,
            num_views=num_views,
        )
        features = torch.from_numpy(self.get_fvd_feats(segments)).double().to(self.device)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += features.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += features.shape[0]
        return int(features.shape[0])

    def _distributed_sync(self):
        if self.is_distributed and dist.is_initialized():
            if torch.cuda.is_available():
                dist.barrier(device_ids=[torch.cuda.current_device()])
            else:
                dist.barrier()
        else:
            return

        for tensor in [self.real_features_sum, self.real_features_cov_sum]:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.real_features_num_samples, op=dist.ReduceOp.SUM)

        for tensor in [self.fake_features_sum, self.fake_features_cov_sum]:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.fake_features_num_samples, op=dist.ReduceOp.SUM)

        self.is_distributed = False

    def compute(self) -> Tensor:
        """Compute one FVD score over the full accumulated validation set."""
        self._distributed_sync()
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one 16-frame segment is required for both real and fake videos to compute FVD")

        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)

        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)

        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(torch.float32)

    def reset(self):
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1
        mx_num_feats = (self.num_features, self.num_features)
        self.real_features_sum = torch.zeros(self.num_features).double().to(self.device)
        self.real_features_cov_sum = torch.zeros(mx_num_feats).double().to(self.device)
        self.real_features_num_samples = torch.tensor(0).long().to(self.device)
        self.fake_features_sum = torch.zeros(self.num_features).double().to(self.device)
        self.fake_features_cov_sum = torch.zeros(mx_num_feats).double().to(self.device)
        self.fake_features_num_samples = torch.tensor(0).long().to(self.device)


class FIDDistMetric:
    pretrained_model_path = os.getenv(
        "FID_MODEL_PATH",
        "models/fid/weights-inception-2015-12-05-6726825d.pth",
    )

    def __init__(
        self,
        num_features: int = 2048,
        reset_real_features: bool = True,
        normalize: bool = False,
        feature_extractor_weights_path: Optional[str] = None,
        device='cpu',
        dtype=torch.float32,
    ) -> None:
        if not isinstance(normalize, bool):
            raise ValueError("Argument `normalize` expected to be a bool")
        self.normalize = normalize
        self.device = device
        self.dtype = dtype
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        valid_int_input = (64, 192, 768, 2048)
        assert num_features in valid_int_input, f"Integer input to argument `feature` must be one of {valid_int_input}"
        feature_extractor_weights_path = (
            feature_extractor_weights_path
            if feature_extractor_weights_path is not None
            else resolve_existing_path(self.pretrained_model_path)
        )

        self.inception = NoTrainInceptionV3(
            name="inception-v3-compat",
            features_list=[str(num_features)],
            feature_extractor_weights_path=feature_extractor_weights_path,
        ).to(self.device)
        self.reset_real_features = reset_real_features
        self.num_features = num_features

        # Initialize metric states.
        mx_num_feats = (num_features, num_features)

        # Real-image statistics.
        self.real_features_sum = torch.zeros(num_features).double().to(device)
        self.real_features_cov_sum = torch.zeros(mx_num_feats).double().to(device)
        self.real_features_num_samples = torch.tensor(0).long().to(device)

        # Generated-image statistics.
        self.fake_features_sum = torch.zeros(num_features).double().to(device)
        self.fake_features_cov_sum = torch.zeros(mx_num_feats).double().to(device)
        self.fake_features_num_samples = torch.tensor(0).long().to(device)

    def _distributed_sync(self):
        """Run explicit distributed synchronization."""
        if self.is_distributed and dist.is_initialized():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            return

        # Synchronize real-image statistics.
        for tensor in [self.real_features_sum, self.real_features_cov_sum]:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.real_features_num_samples, op=dist.ReduceOp.SUM)

        # Synchronize generated-image statistics.
        for tensor in [self.fake_features_sum, self.fake_features_cov_sum]:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(self.fake_features_num_samples, op=dist.ReduceOp.SUM)

        # Prevent double synchronization.
        self.is_distributed = False

    def update(self, imgs: Tensor, real: bool) -> None:
        """Update the state with extracted features."""
        imgs = imgs.detach().to(device=self.device, dtype=self.dtype)
        imgs = (imgs * 255).byte() if self.normalize else imgs
        features = self.inception(imgs)
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)

        # Update the matching state bucket for real or generated images.
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += imgs.shape[0]
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += imgs.shape[0]

    def compute(self) -> Tensor:
        """Calculate FID score with explicit synchronization before computation."""
        # Explicitly synchronize before computing the final metric.
        self._distributed_sync()

        # Ensure both distributions have enough samples.
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError("More than one sample is required for both the real and fake distributed to compute FID")

        # Compute means and covariances.
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(0)
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(0)

        cov_real_num = self.real_features_cov_sum - self.real_features_num_samples * mean_real.t().mm(mean_real)
        cov_real = cov_real_num / (self.real_features_num_samples - 1)

        cov_fake_num = self.fake_features_cov_sum - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)

        return _compute_fid(mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake).to(self.orig_dtype)

    def reset(self):
        """Reset all metric states."""
        # Reset the distributed flag.
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        # Reset real-image statistics.
        self.real_features_sum = torch.zeros(self.num_features).double().to(self.device)
        self.real_features_cov_sum = torch.zeros((self.num_features, self.num_features)).double().to(self.device)
        self.real_features_num_samples = torch.tensor(0).long().to(self.device)

        # Reset generated-image statistics.
        self.fake_features_sum = torch.zeros(self.num_features).double().to(self.device)
        self.fake_features_cov_sum = torch.zeros((self.num_features, self.num_features)).double().to(self.device)
        self.fake_features_num_samples = torch.tensor(0).long().to(self.device)