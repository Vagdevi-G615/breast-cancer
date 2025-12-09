# model_loader.py
import os
import torch
import torch.nn as nn
from torchvision import models

MODEL_FILENAME = "final_best_fast_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_efficientnet_b0(num_classes=3, device=DEVICE):
    """
    Build EfficientNet-B0 and apply the same small modifications
    you used during training: replace first conv and classifier.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # modify first conv to ensure same shape as used in training (you did this in training)
    # The original conv in torchvision has out_channels=32; you replaced the kernel explicitly.
    try:
        model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
    except Exception:
        # if structure differs, try a safe approach: replace only if exists
        pass

    # replace classifier to num_classes
    try:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    except Exception:
        # fallback: if classifier layout differs
        model.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(model.classifier[1].in_features, num_classes))

    return model

def load_model_weights(model, model_path=None, map_location="cpu"):
    """
    Loads weights (supports both plain state_dict and dict with 'state_dict' key).
    """
    if model_path is None:
        model_path = MODEL_FILENAME
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please place it in the repo root or set MODEL_DOWNLOAD_URL to download it.")
    state = torch.load(model_path, map_location=map_location)
    # support wrapper dict
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    # try forgiving load (strip module prefix if present)
    try:
        model.load_state_dict(state)
    except RuntimeError:
        new_state = {}
        for k, v in state.items():
            new_state[k.replace('module.', '')] = v
        model.load_state_dict(new_state)
    return model

def get_model(num_classes=3, model_path=None):
    model = build_efficientnet_b0(num_classes=num_classes)
    try:
        model = load_model_weights(model, model_path=model_path, map_location=DEVICE)
    except Exception as e:
        # raise the exception so the caller knows model couldn't be loaded
        raise e
    model.to(DEVICE)
    model.eval()
    return model
