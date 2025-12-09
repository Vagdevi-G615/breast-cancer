# app.py
import os
import io
from PIL import Image
import gradio as gr
import traceback

from model_loader import get_model, DEVICE
from infer import predict_from_pil, CLASS_NAMES, preprocess
from gradcam_utils import run_gradcam_pp, USE_PYTORCH_GRAD_CAM

MODEL_PATH = "final_breast_cancer_model.pth"
TSNE_PNG = "tsne_plot.png"
# Optional: precomputed heatmaps folder
HEATMAPS_DIR = "static/heatmaps"

# Try load model at startup (if available)
MODEL = None
try:
    MODEL = get_model(num_classes=len(CLASS_NAMES), model_path=MODEL_PATH)
    print("Model loaded in app.")
except Exception as e:
    print("Model could not be loaded at startup:", e)
    MODEL = None

def predict_image(image):
    """
    Tab 1: prediction
    """
    if image is None:
        return "No image", "No confidences"
    try:
        res = predict_from_pil(MODEL, image, device=DEVICE) if MODEL is not None else {"pred_idx": None, "pred_label": "no-model", "confidences": {k: 0.0 for k in CLASS_NAMES}}
        label = res["pred_label"]
        confs = res["confidences"]
        conf_text = "\n".join([f"{k}: {v:.4f}" for k, v in confs.items()])
        return label, conf_text
    except Exception as e:
        traceback.print_exc()
        return f"Error: {e}", ""

def gradcam_tab(image):
    """
    Tab 2: Grad-CAM++ overlay. Prefer precomputed heatmap if exists (fast).
    """
    if image is None:
        return None
    # try precomputed heatmap by filename
    try:
        # If Gradio provides filename, check heatmaps dir
        fname = getattr(image, "filename", None)
        if fname is not None:
            base = os.path.basename(fname)
            maybe = os.path.join(HEATMAPS_DIR, f"{base}_cam.png")
            if os.path.exists(maybe):
                return Image.open(maybe).convert("RGB")
    except Exception:
        pass

    # If not precomputed, run on-demand if library available and model loaded
    if MODEL is None:
        return None

    if not USE_PYTORCH_GRAD_CAM:
        return None

    try:
        # choose top predicted class from the model
        pred = predict_from_pil(MODEL, image, device=DEVICE)
        target_class = pred["pred_idx"]
        # run gradcam
        cam_img = run_gradcam_pp(MODEL, image, preprocess_fn=preprocess, target_class=target_class, use_cuda=(DEVICE=="cuda"))
        return cam_img
    except Exception as e:
        print("Grad-CAM failed:", e)
        return None

def get_tsne_image():
    if os.path.exists(TSNE_PNG):
        return TSNE_PNG
    return None

# Build Gradio UI with tabs
with gr.Blocks() as demo:
    gr.Markdown("# EfficientNet-B0 — Breast Cancer Demo (3-Tab Dashboard)")
    with gr.Tab("Prediction"):
        gr.Markdown("Upload a mammogram image to get model prediction and confidence scores.")
        inp1 = gr.Image(type="pil", label="Upload image")
        pred_label = gr.Textbox(label="Predicted Label")
        pred_conf = gr.Textbox(label="Confidences")
        btn1 = gr.Button("Predict")
        btn1.click(fn=predict_image, inputs=inp1, outputs=[pred_label, pred_conf])

    with gr.Tab("Grad-CAM++"):
        gr.Markdown("Grad-CAM++ overlay. If precomputed heatmaps exist in `static/heatmaps/`, the app will show them (fast). Otherwise the app will compute Grad-CAM++ on demand (may be slow on CPU).")
        inp2 = gr.Image(type="pil", label="Upload image for Grad-CAM++")
        out_cam = gr.Image(type="pil", label="Grad-CAM++ overlay")
        btn2 = gr.Button("Generate Grad-CAM++")
        btn2.click(fn=gradcam_tab, inputs=inp2, outputs=out_cam)

    with gr.Tab("t-SNE latent space"):
        gr.Markdown("t-SNE latent space visualization (static).")
        tsne_img = gr.Image(value=get_tsne_image(), label="t-SNE latent space (precomputed)")

    gr.Markdown("**Notes:**\n- If model is missing the app runs in stub mode (no real predictions). Place `final_best_fast_model.pth` in repo root before pushing to Hugging Face for full functionality.\n- Precompute Grad-CAM++ heatmaps and put them in `static/heatmaps/` as `<original_filename>_cam.png` for instant display.")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)),share=True)
