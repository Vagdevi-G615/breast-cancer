# README.md — EfficientNet-B0 Breast Cancer Classifier (Grad-CAM++ + t-SNE Dashboard)

## 🚀 Overview

This project is an end-to-end breast cancer image classification system built using EfficientNet-B0, enhanced with:

* Grad-CAM++ Explainability
* t-SNE Latent Space Visualization
* Interactive Gradio Dashboard

The model classifies mammogram patches into three categories:

* 0 → Benign
* 1 → Malignant
* 2 → Normal

##  Model Summary

* Backbone: EfficientNet-B0 (torchvision)
* Pretrained weights: `IMAGENET1K_V1`
* Modified first convolution layer
* Custom classifier for 3 classes
* Trained on cleaned mammogram dataset
* Saved as `final_breast_cancer_model.pth`

---

## 🔍 Features

### Image Classification

Upload an image → Model predicts class + confidence scores.

### Grad-CAM++ Explainability

Generates heatmaps that highlight important cancerous regions.

### t-SNE Visualization

Displays the latent-space feature clusters of your dataset.

### Fully Interactive Gradio UI

Three dashboard tabs:

1. Prediction
2. Grad-CAM++
3. t-SNE Plot

---

## 📂 Project Structure

```
repo/
├── app.py                 # Main Gradio application
├── model_loader.py        # EfficientNet model reconstruction + weight loader
├── gradcam_utils.py       # Grad-CAM++ implementation
├── infer.py               # Preprocessing + prediction utilities
├── final_best_fast_model.pth
├── tsne_plot.png          # Saved t-SNE plot
├── requirements.txt
└── README.md
```

---

##  Running Locally

### 1. Install dependencies

```
pip install -r requirements.txt
```

### 2. Launch the Gradio app

```
python app.py
```

Open in browser:

```
http://127.0.0.1:7860
```

---

## Deploying to Hugging Face Spaces

### 1. Create a Space

* Go to: [https://huggingface.co/spaces](https://huggingface.co/spaces)
* New Space → Gradio → CPU Basic

### 2. Upload or connect GitHub repository

### 3. Ensure the following files are in the root:

* `app.py`
* `model_loader.py`
* `gradcam_utils.py`
* `infer.py`
* `final_best_fast_model.pth`
* `requirements.txt`
* `tsne_plot.png`

Hugging Face will automatically build and serve your application.

---
✅ Badges
✅ HuggingFace Space preview screenshot
✅ WandB logging section
✅ Citation / BibTeX

Just tell me and I’ll add it.
