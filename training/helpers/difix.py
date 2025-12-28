from typing import Any, Dict, Tuple

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from submodules.difix3d.src.pipeline_difix import DifixPipeline


def tensor_to_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert a [H,W,C] tensor in [0,1] to uint8 numpy."""
    return (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")

def pad_image_to_multiple(image: np.ndarray, multiple: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    pad_h = (multiple - (image.shape[0] % multiple)) % multiple
    pad_w = (multiple - (image.shape[1] % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return padded, (pad_h, pad_w)

def prepare_image_for_difix(image: np.ndarray, resolution_multiple: int = 8) -> Tuple[np.ndarray, Dict[str, Any]]:
    multiple = max(1, resolution_multiple)
    metadata: Dict[str, Any] = {
        "orig_hw": image.shape[:2],
    }
    processed, pad_hw = pad_image_to_multiple(image, multiple)
    metadata["pad_hw"] = pad_hw
    metadata["processed_hw"] = processed.shape[:2]
    return processed, metadata

def restore_image_from_difix(image: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
    orig_h, orig_w = metadata.get("orig_hw", image.shape[:2])
    return image[:orig_h, :orig_w, :]

@torch.no_grad()
def difix_refine(difix_cfg, renders: torch.Tensor, reference_images: torch.Tensor, difix_pipe: DifixPipeline, is_eval=False, device="cuda") -> torch.Tensor:
    """Refine rendered images using Difix with reference images.
    Args:
        renders: [B, H, W, 3] rendered images to refine
        reference_images: [B, H, W, 3] reference images for refinement
        difix_pipe: DifixPipeline instance
    Returns:
        refined_renders: [B, H, W, 3] refined rendered images
    """
    if difix_pipe is None:
        return renders

    # - Run the refinement
    refined_renders = []
    for i in tqdm(range(renders.shape[0]), desc="Difix refinement", total=renders.shape[0], leave=False):
        # -- Img to refine
        img_to_refine = tensor_to_uint8(renders[i])
        # -- Reference image
        reference_image = tensor_to_uint8(reference_images[i])

        img_prepared, transform_meta = prepare_image_for_difix(img_to_refine)
        ref_prepared, _ = prepare_image_for_difix(reference_image)

        # -- Run Difix
        if not is_eval:
            refined_image = difix_pipe(
                difix_cfg.prompt,
                image=Image.fromarray(img_prepared),
                ref_image=Image.fromarray(ref_prepared),
                height=img_prepared.shape[0],
                width=img_prepared.shape[1],
                num_inference_steps=difix_cfg.num_inference_steps,
                timesteps=difix_cfg.timesteps,
                guidance_scale=difix_cfg.guidance_scale,
                negative_prompt=difix_cfg.negative_prompt,
            ).images[0]
        # (the only difference in eval is that we do not provide ref image)
        else:
            refined_image = difix_pipe(
                difix_cfg.prompt,
                image=Image.fromarray(img_prepared),
                height=img_prepared.shape[0],
                width=img_prepared.shape[1],
                num_inference_steps=difix_cfg.num_inference_steps,
                timesteps=difix_cfg.timesteps,
                guidance_scale=difix_cfg.guidance_scale,
                negative_prompt=difix_cfg.negative_prompt,
            ).images[0]

        # -- Collect results
        refined_np = np.array(refined_image)
        refined_np = restore_image_from_difix(refined_np, transform_meta)
        refined_tensor = torch.from_numpy(refined_np).float() / 255.0
        assert (
            refined_tensor.shape == renders[i].shape
        ), "Refined image has different shape than the original render after restoration."
        refined_renders.append(refined_tensor.to(device))

    # - Stack refined renders
    refined_renders = torch.stack(refined_renders, dim=0) # [B, H, W, 3]

    return refined_renders
