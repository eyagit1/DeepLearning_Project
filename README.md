# 🔬 Cancer Detection using Vision Transformers (ViT-B/16)

A deep learning project for histopathological cancer classification using **Vision Transformer (ViT-Base/16)** with explainability. The project contains two notebooks — one for colon cancer (binary classification) and one for lung cancer (3-class classification) — both built on the same modular pipeline.

---

## 📁 Project Structure

```
├── colon_cancer_ViT_final.ipynb        # Binary classification: Adenocarcinoma vs Benign
└── lung_cancer_ViT_final.ipynb         # 3-class: Adenocarcinoma / Squamous Cell / Benign
```

---

## 🎯 Tasks

### Colon Cancer Notebook
Binary classification of colon histopathology images:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | `colon_aca` | Colon Adenocarcinoma (malignant) |
| 1 | `colon_n` | Colon Benign Tissue (healthy) |

### Lung Cancer Notebook
3-class classification of lung histopathology images:

| Class | Label | Description |
|-------|-------|-------------|
| 0 | `lung_n` | Benign (healthy) lung tissue |
| 1 | `lung_aca` | Lung Adenocarcinoma |
| 2 | `lung_scc` | Lung Squamous Cell Carcinoma |

---

## 🗃️ Dataset

**[LC25000 — Lung and Colon Histopathological Image Dataset](https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)**

- 5,000 images per class at 768×768px
- Resized to 224×224 for ViT input
- Available on Kaggle

---

## 🏗️ Model Architecture

**ViT-Base/16** pretrained on ImageNet-21k, fine-tuned via transfer learning.

```
Input Image (224×224)
      ↓  split into 196 patches of 16×16 px
Patch Embedding  →  196 + 1 CLS token = 197 tokens
      ↓
12× Transformer Encoder Blocks
   ├── Multi-Head Self-Attention (12 heads)
   └── Feed-Forward Network (MLP)
      ↓
CLS Token  →  Linear Head  →  2 or 3 classes
```

- **Total parameters**: ~86M
- **Classification head**: `Linear(768 → 2)` for colon, `Linear(768 → 3)` for lung
- Optional backbone freezing: train only the last 4 Transformer blocks + head for small datasets

---

## ⚙️ Training Configuration

| Parameter | Colon | Lung |
|-----------|-------|------|
| `IMG_SIZE` | 224 | 224 |
| `BATCH_SIZE` | 32 | 32 |
| `NUM_EPOCHS` | 10 | 15 |
| `LR` | 2e-4 | 2e-4 |
| `WEIGHT_DECAY` | 1e-4 | 1e-4 |
| Train / Val / Test split | 70 / 15 / 15% | 70 / 15 / 15% |
| Optimizer | AdamW | AdamW |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Loss | CrossEntropyLoss | Weighted CrossEntropyLoss |

---

## 🖼️ Data Augmentation

Applied to the **training set only**. Validation and test sets use clean transforms (resize + normalize).

| Augmentation | Purpose |
|---|---|
| Random Horizontal & Vertical Flip | Histopathology has no canonical orientation |
| Random Rotation (±15°) | Slide orientation variation |
| Color Jitter (brightness, contrast, saturation) | Stain variation across labs |
| ImageNet Normalization | Required for pretrained ViT |

---

## 🔍 Explainability (XAI)

Both notebooks include explainability methods to make predictions interpretable for clinical use.

### Attention Rollout
Propagates self-attention weights through all 12 transformer layers to identify which image patches the model focused on. Returns a 14×14 attention map upsampled to the original image size.

### Grad-CAM for ViT *(lung notebook only)*
Gradient-weighted Class Activation Mapping adapted for Vision Transformers. Unlike Attention Rollout, Grad-CAM is **class-discriminative** — it shows what the model attends to specifically for the predicted class. Both methods are displayed side-by-side for comparison.

---

## 📊 Pipeline Steps

Both notebooks follow the same step-by-step structure:

| Step | Description |
|------|-------------|
| 0 | GPU Setup & Verification |
| 1 | Library Installation |
| 2 | Dataset Mount & Verification |
| 3 | Imports & Hyperparameter Configuration |
| 4 | Data Augmentation & Dataset Preparation |
| 5 | DataLoaders + Sample Visualization |
| 6 | Pretrained ViT-B/16 Model Loading |
| 7 | Loss Function, Optimizer & LR Scheduler |
| 8 | Training Loop with Best-Model Checkpointing |
| 9 | Training Curves Visualization |
| 10 | Test Set Evaluation + Classification Report |
| 11 | Confusion Matrix |
| 12 | XAI — Attention Rollout (+ Grad-CAM & comparison in lung notebook) |
| 13 | Model Saving & Download *(+ Multi-Modal Fusion Blueprint in lung notebook)* |
| 14 | Single-Image Inference *(+ Multi-Organ Extension in lung notebook)* |

---

## 🏗️ Advanced Features (Lung Notebook)

### Multi-Modal Fusion Blueprint
A ready-to-use architecture blueprint combining three input modalities:

```
Histopathology (224×224)  →  ViT-B/16  (768 dim)  ──┐
CT / X-Ray (224×224)      →  ViT-S/16  (384 dim)  ──├──► Fusion MLP → 3 classes
Clinical Metadata (8 dim) →  MLP Encoder (128 dim) ──┘
```

### Multi-Organ Extension
A configuration registry (`MULTI_ORGAN_CONFIG`) allows the same pipeline to be reused for new organs (colon, breast, prostate, etc.) by simply registering a new entry — no code changes required.

---

## 📦 Requirements

```
Python       >= 3.9
PyTorch      >= 2.0
timm         (latest)
scikit-learn
matplotlib
seaborn
Pillow
numpy
```

Install dependencies:
```bash
pip install timm scikit-learn matplotlib seaborn
```

> PyTorch, numpy, and Pillow are pre-installed on Kaggle and most Google Colab environments.

---

## 🚀 Getting Started

### On Kaggle
1. Open the notebook and go to **Settings → Accelerator → GPU T4 x2 → Save**
2. Click **+ Add Data** and add the LC25000 dataset
3. Run all cells from top to bottom

### On Google Colab
1. Mount your Google Drive in Step 2
2. Set `DATA_DIR` to your dataset folder path in Drive
3. Run all cells — the notebook will use your Drive data directly

---

## 💾 Output Files

After training, the following files are saved to the output directory:

| File | Description |
|------|-------------|
| `best_colon_vit.pth` / `best_lung_vit.pth` | Best checkpoint (saved at peak validation accuracy) |
| `colon_cancer_vit_final.pth` / `lung_cancer_vit_final.pth` | Final checkpoint with full metadata |
| `training_curves.png` | Loss and accuracy curves per epoch |
| `confusion_matrix.png` | Confusion matrix on the test set |
| `attention_heatmaps.png` | Attention Rollout overlays |
| `xai_comparison.png` | Side-by-side Attention Rollout vs Grad-CAM *(lung only)* |
| `metrics_colon.json` / `metrics_lung.json` | Best val accuracy and test accuracy |

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correctness on the test set |
| Precision | Of all predicted positives, how many were correct |
| Recall (Sensitivity) | Of all actual positives, how many were detected — most critical in medical imaging |
| F1-Score | Harmonic mean of Precision and Recall |
| Macro F1 | Average F1 across all classes — primary metric for the 3-class lung task |

> In a medical context, **Recall for the cancer class** is the most important metric — a missed cancer (false negative) is more dangerous than a false alarm.

---

## 🔄 Reproducibility

All random seeds are fixed at `SEED = 42` via `torch.manual_seed` and `np.random.seed`. The train/val/test split uses a fixed `torch.Generator` seed, ensuring the same split across runs.
