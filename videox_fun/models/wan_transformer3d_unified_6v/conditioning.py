# Conditioning modules for Wan Transformer

from functools import partial
from typing import Callable, List, Type, Union

import torch
import torch.nn as nn
from diffusers.models.controlnet import zero_module
from einops import rearrange

from .embeddings import sinusoidal_embedding_batchwise
from ..wan_camera_adapter import SimpleAdapter


class GlobalRepresentationEncoder(nn.Module):
    "UniCeption Global Representation Encoder"

    def __init__(
        self,
        name: str,
        in_chans: int = 3,
        enc_embed_dim: int = 1024,
        intermediate_dims: List[int] = [128, 256, 512],
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-6),
        *args,
        **kwargs,
    ):
        """
        Global Representation Encoder for projecting a global representation to a desired latent dimension.

        Args:
            name (str): Name of the Encoder.
            in_chans (int): Number of input channels.
            enc_embed_dim (int): Embedding dimension of the encoder.
            intermediate_dims (List[int]): List of intermediate dimensions of the encoder.
            act_layer (Type[nn.Module]): Activation layer to use in the encoder.
            norm_layer (Union[Type[nn.Module], Callable[..., nn.Module]]): Final normalization layer to use in the encoder.
        """
        super().__init__(*args, **kwargs)

        # Initialize the attributes
        self.name = name
        self.in_chans = in_chans
        self.enc_embed_dim = enc_embed_dim
        self.intermediate_dims = intermediate_dims

        # Init the activation layer
        self.act_layer = act_layer()

        # Initialize the encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.in_chans, self.intermediate_dims[0]),
            self.act_layer,
        )
        for intermediate_idx in range(1, len(self.intermediate_dims)):
            self.encoder = nn.Sequential(
                self.encoder,
                nn.Linear(self.intermediate_dims[intermediate_idx - 1], self.intermediate_dims[intermediate_idx]),
                self.act_layer,
            )
        self.encoder = nn.Sequential(
            self.encoder,
            nn.Linear(self.intermediate_dims[-1], self.enc_embed_dim),
        )
        final_linear_layer = self.encoder[-1]
        nn.init.zeros_(final_linear_layer.weight)
        nn.init.zeros_(final_linear_layer.bias)

        # Init weights of the final norm layer
        self.norm_layer = norm_layer(enc_embed_dim) if norm_layer else nn.Identity()
        if isinstance(self.norm_layer, nn.LayerNorm):
            nn.init.constant_(self.norm_layer.bias, 0)
            nn.init.constant_(self.norm_layer.weight, 1.0)

    def forward(self, encoder_input):
        """
        Global Representation Encoder Forward Pass

        Args:
            encoder_input: Input data for the encoder.
                The provided data must contain a tensor of size (B, C).

        Returns:
            Output features from the encoder.
        """
        # Get the input data and verify the shape of the input
        input_data = encoder_input.data
        assert input_data.ndim == 2, "Input data must have shape (B, C)"
        assert input_data.shape[1] == self.in_chans, f"Input data must have {self.in_chans} channels"

        # Encode the global representation
        features = self.encoder(input_data)

        # Normalize the output
        features = self.norm_layer(features)

        return features


class PoseEmbeder(nn.Module):
    def __init__(self, in_dims=[16, 12, 4], emb_dim=2048):
        super().__init__()
        self.rot_emb = GlobalRepresentationEncoder(in_chans=in_dims[0], enc_embed_dim=emb_dim, name="rot_enc")
        self.trans_emb = GlobalRepresentationEncoder(in_chans=in_dims[1], enc_embed_dim=emb_dim, name="trans_enc")
        self.scale_emb = GlobalRepresentationEncoder(in_chans=in_dims[2], enc_embed_dim=emb_dim, name="scale_enc")

    def forward(self, x):
        # x: (B, F, 32)
        rot, trans, scale = x.split([16, 12, 4], dim=-1)
        rot_emb = self.rot_emb(rot)
        trans_emb = self.trans_emb(trans)
        scale_emb = self.scale_emb(scale)
        return rot_emb + trans_emb + scale_emb


class MLPProj(torch.nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = torch.nn.Sequential(
            torch.nn.LayerNorm(in_dim), torch.nn.Linear(in_dim, in_dim),
            torch.nn.GELU(), torch.nn.Linear(in_dim, out_dim),
            torch.nn.LayerNorm(out_dim))

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens


class ImageConditioningProj(nn.Module):
    def __init__(self, in_dim, dim, patch_size):
        super().__init__()
        self.conv = nn.Conv3d(
            in_dim, dim,
            kernel_size=patch_size, stride=patch_size)
        self.proj = zero_module(nn.Linear(dim, dim))
        nn.init.xavier_uniform_(self.conv.weight.flatten(1))

    def forward(self, x, y, **kwargs):
        y = self.conv(y)
        y = self.proj(y.flatten(2).transpose(1, 2))
        x = [u + v for u, v in zip(x, y)]
        return x

class ActionConditioningProj(nn.Module):
    def __init__(self, freq_dim, dim, patch_size):
        super().__init__()
        self.action_embedding = nn.Sequential(
            nn.Linear(freq_dim * 3, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        nn.init.normal_(self.action_projection[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.action_projection[-1].bias)
        self.freq_dim = freq_dim
        self.dim = dim

    def forward(self, actions, **kwargs):
        actions = actions * torch.tensor([500.0, 60000.0, 300.0], device=actions.device, dtype=actions.dtype)
        a_emb = sinusoidal_embedding_batchwise(self.freq_dim, actions).float()
        a_emb = a_emb.flatten(-2)  # -> [B, F, 2*freq_dim]
        a = self.action_embedding(a_emb)  # [B, F, dim]
        a0 = self.action_projection(a).unflatten(-1, (6, self.dim))  # [B, F, 6, dim]
        return a0


class PoseAdapter(nn.Module):
    def __init__(self, in_dim, dim, patch_size):
        super().__init__()
        self.adapter = SimpleAdapter(
                in_dim,
                dim,
                kernel_size=patch_size[1:],
                stride=patch_size[1:]
            )
        self.proj = zero_module(nn.Linear(dim, dim))
        nn.init.xavier_uniform_(self.proj.weight.flatten(1))

    def forward(self, x, y, **kwargs):
        y = self.adapter(y)
        y = self.proj(y.flatten(2).transpose(1, 2))
        x = [u + v for u, v in zip(x, y)]
        return x


class ViewIDEmbedding(nn.Module):
    def __init__(self, freq_dim, dim):
        super().__init__()
        self.view_embedding = nn.Linear(freq_dim, dim)
        self.view_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim))

        nn.init.normal_(self.view_projection[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.view_projection[-1].bias)
        self.freq_dim = freq_dim
        self.dim = dim

    def forward(self, x, y, **kwargs):
        num_views = kwargs['num_views']
        device = kwargs["device"]
        dtype = kwargs.get("dtype", torch.float32)
        # 1, nf*h*w, d
        nfhw = x[0].shape[1]
        fhw = nfhw // num_views

        view_ids = y.to(device).float() * 100.0
        views_emb = sinusoidal_embedding_batchwise(self.freq_dim, view_ids).to(dtype)  # (n, d)

        views_emb = self.view_embedding(views_emb)  # (n, d)
        views_emb = self.view_projection(views_emb)  # (n, d)

        views_emb = views_emb.unsqueeze(1).repeat(1, fhw, 1)
        views_emb = rearrange(views_emb, 'n s d -> (n s) d', n=num_views, s=fhw)
        x = [u + views_emb.unsqueeze(0) for u in x]  # (1, nfhw, d)
        return x


class LearnbleViewIDEmb(nn.Module):
    def __init__(self, max_num_views, dim):
        super().__init__()
        self.view_embedding = nn.Embedding(max_num_views, dim)

        self.view_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        # init
        nn.init.normal_(self.view_embedding.weight, std=0.02)
        nn.init.normal_(self.view_projection[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.view_projection[-1].bias)
        self.dim = dim

    def forward(self, x, y, **kwargs):
        """
        Args:
            x: List of input tensors
            y: view_ids tensor. Shape should be [current_num_views]
            kwargs: contains 'device', 'dtype'. 'num_views' is optional/redundant here if we trust y.
        """
        device = kwargs["device"]
        dtype = kwargs.get("dtype", torch.float32)
        
        # y contains view indices. Ensure they are Long for embedding lookup
        view_ids = y.to(device).long() 
        
        # 获取当前实际处理的 num_views (n)，而不是依赖 kwargs 里的值
        # 这样即使 num_views != max_num_views 也能正确处理
        current_num_views = view_ids.shape[0]
        
        # x[0] shape is [1, nf*h*w, d]
        nfhw = x[0].shape[1]
        
        # 计算每个 view 的空间-时间 token 数量
        fhw = nfhw // current_num_views

        # Lookup learnable embeddings [current_num_views, dim]
        # 这里会根据传入的 id (如 [0, 2]) 取出对应的 embedding
        views_emb = self.view_embedding(view_ids)
        
        # Project [current_num_views, dim] -> [current_num_views, dim]
        views_emb = self.view_projection(views_emb).to(dtype)
        
        # Broadcast to spatial-temporal dimensions
        # Expand: [current_num_views, dim] -> [current_num_views, fhw, dim]
        views_emb = views_emb.unsqueeze(1).repeat(1, fhw, 1)
        
        # Flatten: [current_num_views, fhw, dim] -> [nfhw, dim]
        # 使用 current_num_views 确保 rearrange 维度正确
        views_emb = rearrange(views_emb, 'n s d -> (n s) d', n=current_num_views, s=fhw)
        
        # Add to input [1, nfhw, dim]
        x = [u + views_emb.unsqueeze(0) for u in x]
        return x
