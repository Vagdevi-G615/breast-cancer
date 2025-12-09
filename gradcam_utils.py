# gradcam_utils.py
import os
import numpy as np
from PIL import Image
import torch

USE_PYTORCH_GRAD_CAM = False
try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    USE_PYTORCH_GRAD_CAM = True
except Exception:
    USE_PYTORCH_GRAD_CAM = False

def find_target_layer(model):
    """
    Heuristic to find the last good convolutional layer for EfficientNet-B0.
    """
    # Common target for torchvision EfficientNet: model.features[-1] or model.features[-1][-1]
    target = None
    try:
        # Try the last block's last conv
        feats = list(model.features.children())
        if len(feats) > 0:
            last_block = feats[-1]
            # drill down if nested
            if hasattr(last_block, 'conv'):
                target = last_block.conv
            else:
                # find last conv-like module inside last_block
                for module in reversed(list(last_block.modules())):
                    if isinstance(module, torch.nn.Conv2d):
                        target = module
                        break
    except Exception:
        target = None

    if target is None:
        # fallback to scanning for last Conv2d in whole model
        for module in reversed(list(model.modules())):
            if isinstance(module, torch.nn.Conv2d):
                target = module
                break
    return target

def run_gradcam_pp(model, pil_image, preprocess_fn, target_class=None, use_cuda=False):
    """
    Run Grad-CAM++ and return a PIL image of the overlay (RGB).
    - model: torch model (eval)
    - pil_image: PIL.Image
    - preprocess_fn: function that converts PIL->tensor (including normalization)
    - target_class: int or None
    - use_cuda: whether to run on GPU
    """
    if not USE_PYTORCH_GRAD_CAM:
        raise RuntimeError("pytorch-grad-cam not installed or unavailable. See requirements.txt for installation.")

    device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
    model.to(device)

    target_layer = find_target_layer(model)
    if target_layer is None:
        raise RuntimeError("Could not find a target conv layer for Grad-CAM++.")

    input_tensor = preprocess_fn(pil_image).unsqueeze(0).to(device)

    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer], use_cuda=(device == "cuda"))
    targets = None if target_class is None else [ClassifierOutputTarget(int(target_class))]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (H,W)

    # Convert original image to float numpy in range [0,1]
    img_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
    cam_overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
    return Image.fromarray(cam_overlay)
