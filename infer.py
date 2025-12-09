# infer.py
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms

# Class labels (map indices to names)
CLASS_NAMES = ["benign", "malignant", "normal"]

IMG_SIZE = 224  # change if your training used a different size

# Preprocessing used during training (ImageNet-like)
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

def predict_from_pil(model, pil_img, device="cpu"):
    """
    Returns: dict with 'pred_idx', 'pred_label', 'confidences' (dict)
    """
    model.to(device)
    model.eval()
    img = pil_img.convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    confidences = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    return {"pred_idx": pred_idx, "pred_label": CLASS_NAMES[pred_idx], "confidences": confidences}
