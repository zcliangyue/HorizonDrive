import os
import inspect
import importlib
import numpy as np
import torch
import cv2
from einops import rearrange
from PIL import Image
import imageio


def filter_kwargs(cls, kwargs):
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs


def get_multiview_tile_img(
    videos,
    camera_names=None,
    fid_metric=None,
    fvd_metric=None,
    rescale=False,
):
    videos = rearrange(videos, "(n t) c h w -> t n c h w", n=len(camera_names))
    outputs = []
    for i, x in enumerate(videos):
        if rescale:
            x = (x + 1.0) / 2.0
        x = x.clamp(0.0, 1.0)
        x = (x.float() * 255).detach().cpu().numpy().astype(np.uint8)   # (n,c,h,w)
        fvd_str = ""
        channel, landscape_height, landscape_width = x.shape[1], x.shape[2], x.shape[3]
        height = landscape_height * 3
        width = landscape_width * 3
        tiled_img = np.zeros((height, width, channel), dtype=np.uint8)
        for cam_id, cam_name in enumerate(camera_names):
            cam_img = x[cam_id].transpose(1, 2, 0)
            cam_img = np.ascontiguousarray(cam_img)
            if fid_metric is not None:
                cv2.putText(cam_img, f"FID: {fid_metric[cam_id]:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if fvd_metric is not None:
                fvd_str += f"{fvd_metric[cam_id]:.2f} "

            if len(camera_names) == 1:
                tiled_img = cam_img
                height, width = landscape_height, landscape_width
                break

            if cam_name == "camera_front_30fov":
                # Place CAM_FRONT_30FOV at the top center
                tiled_img[:landscape_height, landscape_width:2*landscape_width, :] = cam_img
            elif cam_name == "camera_front_left":
                # Place CAM_FRONT_LEFT at the left center
                tiled_img[landscape_height:2*landscape_height, :landscape_width, :] = cam_img
            elif cam_name == "camera_front":
                # Place CAM_FRONT at the center
                tiled_img[landscape_height:2*landscape_height, landscape_width:2*landscape_width, :] = cam_img
            elif cam_name == "camera_front_right":
                # Place CAM_FRONT_RIGHT at the right center
                tiled_img[landscape_height:2*landscape_height, 2*landscape_width:, :] = cam_img
            elif cam_name == "camera_rear_left":
                # Place CAM_BACK_LEFT at the bottom left
                tiled_img[2*landscape_height:, :landscape_width, :] = cam_img
            elif cam_name == "camera_rear":
                # Place CAM_BACK at the bottom center
                tiled_img[2*landscape_height:, landscape_width:2*landscape_width, :] = cam_img
            elif cam_name == "camera_rear_right":
                # Place CAM_BACK_RIGHT at the bottom right
                tiled_img[2*landscape_height:, 2*landscape_width:, :] = cam_img

        if fvd_metric is not None:
            fvd_str = f"FVD: {fvd_str}"
            cv2.putText(tiled_img, fvd_str, (width // 2 - 100, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        outputs.append(tiled_img)
    return outputs


def save_multiview_videos_grid(
    videos: torch.Tensor,
    path: str,
    rescale=False,
    gt_videos: torch.Tensor= None,
    fps=10,
    imageio_backend=True,
    color_transfer_post_process=False,
    camera_names=None,
    fid_metric=None,
    fvd_metric=None,
    overlap=False,
):
    """Combine cameras into a tiled image.
    Layout:
        ################################################################
        #                 #  CAM_FRONT_30FOV   #                       #
        ################################################################
        # CAM_FRONT_LEFT  #     CAM_FRONT      #     CAM_FRONT_RIGHT   #
        ################################################################
        #  CAM_BACK_LEFT  #     CAM_BACK       #     CAM_BACK_RIGHT    #
        ################################################################
    """
    videos_outputs = get_multiview_tile_img(
        videos,
        camera_names=camera_names,
        fid_metric=fid_metric,
        fvd_metric=fvd_metric,
        rescale=rescale,
    )
    if gt_videos is not None:
        gt_videos_outputs = get_multiview_tile_img(
            gt_videos,
            camera_names=camera_names,
            rescale=rescale,
        )
    outputs = []
    for i, tiled_img in enumerate(videos_outputs):
        if gt_videos is not None:
            gt_tiled_img = gt_videos_outputs[min(i, len(gt_videos_outputs) - 1)]
            if not overlap:
                tiled_img = np.concatenate([tiled_img, np.zeros((16, tiled_img.shape[1], tiled_img.shape[2]), dtype=np.uint8)], axis=0)
                tiled_img = np.concatenate([tiled_img, gt_tiled_img], axis=0)
            else:
                condition_mask = np.max(tiled_img, axis=-1, keepdims=True) > 8
                condition_overlay = cv2.addWeighted(tiled_img, 0.75, gt_tiled_img, 0.25, 0)
                tiled_img = np.where(condition_mask, condition_overlay, gt_tiled_img)
        outputs.append(Image.fromarray(tiled_img))

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs,quality=8,codec='libx264', macro_block_size=16,fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1/fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)


def import_cls(type):
    """
    Import class based on its full path.

    Args:
        type (str): Full path of the class, e.g., 'module.submodule.ClassName'.

    Returns:
        class: The imported class.
    """
    module, cls = type.rsplit('.', 1)
    module = importlib.import_module(module, package=None)
    cls = getattr(module, cls)
    return cls


def construct_emb_cls(emb_info, additional_kwargs={}):
    """
    Construct embedding class based on provided information.

    Args:
        emb_info (dict): Information about the embedding, including 'name' and 'type'.
        additional_kwargs(dict): Additional arguments for the embedding class.

    Returns:
        nn.Module: An instance of the specified embedding class.
    """
    emb_cls = import_cls(emb_info['type'])
    all_kwargs = emb_info.get('kwargs', {})
    all_kwargs.update(additional_kwargs)
    return emb_cls(**all_kwargs)