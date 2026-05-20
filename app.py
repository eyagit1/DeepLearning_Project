import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os


import gdown
import os

MODEL_FILES = {
    "models/colon_cancer_vit_final.pth": "1MYumugVfVmvyWc453TJ4z-3sUHMIfTry",
    "models/lung_cancer_vit_final.pth":  "12RAqhwTuRAgiiOJn3gBwP7CH7TRX4gQb",
}

os.makedirs("models", exist_ok=True)
for path, file_id in MODEL_FILES.items():
    if not os.path.exists(path):
        with st.spinner(f"Downloading {os.path.basename(path)}..."):
            gdown.download(id=file_id, output=path, quiet=False)

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="CancerDetect AI - Histopathology Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e40af;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .malignant-box {
        background-color: #fef2f2;
        border-left-color: #dc2626;
    }
    .benign-box {
        background-color: #f0fdf4;
        border-left-color: #16a34a;
    }
    .info-box {
        background-color: #eff6ff;
        border-left-color: #2563eb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
        .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# XAI CLASSES
# ============================================
class ViTGradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.activations = None
        self.gradients = None
        layer = target_layer if target_layer else model.blocks[-1]
        self._fwd = layer.register_forward_hook(self._save_act)
        self._bwd = layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_tensor, device, class_idx=None):
        self.model.eval()
        img_t = img_tensor.unsqueeze(0).to(device)
        img_t.requires_grad_(True)
        output = self.model(img_t)
        pred_class = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1).max().item()
        target = class_idx if class_idx is not None else pred_class
        self.model.zero_grad()
        output[0, target].backward()
        weights = self.gradients[0, 1:].mean(dim=-1)
        activations = self.activations[0, 1:]
        cam = F.relu((weights.unsqueeze(-1) * activations).sum(-1))
        gs = int(cam.shape[0] ** 0.5)
        cam = cam.reshape(gs, gs).cpu().detach().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)
        ) / 255.0
        return cam, pred_class, confidence

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()


def compute_attention_rollout(model, img_tensor, device, head_fusion='mean', discard_ratio=0.9):
    model.eval()
    attention_maps = []

    def hook_fn(module, inp, out):
        B, N, C = inp[0].shape
        qkv = module.qkv(inp[0]).reshape(
            B, N, 3, module.num_heads, C // module.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, _ = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * (C // module.num_heads) ** -0.5
        attn = attn.softmax(dim=-1).detach().cpu()
        attention_maps.append(attn)

    hooks = [block.attn.register_forward_hook(hook_fn) for block in model.blocks]

    with torch.no_grad():
        out = model(img_tensor.unsqueeze(0).to(device))
        pred_class = out.argmax(1).item()
        confidence = torch.softmax(out, dim=1).max().item()

    for h in hooks:
        h.remove()

    rollout = torch.eye(attention_maps[0].shape[-1])
    for attn in attention_maps:
        if head_fusion == 'mean':
            fused = attn.mean(1)[0]
        elif head_fusion == 'min':
            fused = attn.min(1).values[0]
        else:
            fused = attn.max(1).values[0]
        thresh = torch.quantile(fused.view(-1), discard_ratio)
        fused[fused < thresh] = 0
        fused = fused + torch.eye(fused.shape[0])
        fused = fused / fused.sum(dim=-1, keepdim=True)
        rollout = fused @ rollout

    mask = rollout[0, 1:]
    gs = int(mask.shape[0] ** 0.5)
    mask = mask.reshape(gs, gs).numpy()
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask, pred_class, confidence


# ============================================
# MODEL LOADING — loads your .pth checkpoints
# ============================================
MODEL_PATHS = {
    'lung':  'models/lung_cancer_vit_final.pth',
    'colon': 'models/colon_cancer_vit_final.pth',
}

@st.cache_resource
def load_model(task: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = MODEL_PATHS[task]

    if not os.path.exists(path):
        st.error(f"❌ Model file not found: `{path}`\n\nMake sure you placed your trained `.pth` files inside a `models/` folder next to `app.py`.")
        st.stop()

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']
    config      = checkpoint.get('config', {})

    model = timm.create_model(
        config.get('model', 'vit_base_patch16_224'),
        pretrained=False,
        num_classes=num_classes
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    model_info = {
        'Test Accuracy': f"{checkpoint.get('test_accuracy', 0) * 100:.2f}%",
        'Val Accuracy':  f"{checkpoint.get('best_val_accuracy', 0) * 100:.2f}%",
        'Classes':       str(num_classes),
        'Architecture':  config.get('model', 'vit_base_patch16_224'),
    }

    return model, device, class_names, model_info


# ============================================
# IMAGE PREPROCESSING
# ============================================
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image)

def denormalize_image(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img  = tensor * std + mean
    return np.clip(img.permute(1, 2, 0).numpy(), 0, 1)


# ============================================
# VISUALIZATION
# ============================================
def create_xai_comparison(img_np, attention_mask, grad_cam_mask, class_names, pred_class):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np);  axes[0].set_title('Original Image', fontsize=14, fontweight='bold');  axes[0].axis('off')
    attention_up = np.array(Image.fromarray((attention_mask * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
    axes[1].imshow(img_np);  axes[1].imshow(attention_up, alpha=0.5, cmap='jet');  axes[1].set_title('Attention Rollout', fontsize=14, fontweight='bold');  axes[1].axis('off')
    axes[2].imshow(img_np);  axes[2].imshow(grad_cam_mask, alpha=0.5, cmap='jet'); axes[2].set_title(f'Grad-CAM: {class_names[pred_class]}', fontsize=14, fontweight='bold'); axes[2].axis('off')
    plt.tight_layout()
    return fig

def plot_confidence_scores(probs, class_names):
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#16a34a', '#dc2626', '#f59e0b'][:len(class_names)]
    bars = ax.barh(class_names, probs, color=colors, alpha=0.8)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence', fontsize=12, fontweight='bold')
    ax.set_title('Classification Confidence', fontsize=14, fontweight='bold')
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', ha='left', va='center', fontweight='bold')
    plt.tight_layout()
    return fig


# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.markdown("# 🔬 CancerDetect AI")
    st.markdown("---")

    st.markdown("### Model Selection")
    task = st.radio(
        "Select Classification Task:",
        ["🫁 Lung Cancer (3-class)", "🔬 Colon Cancer (2-class)"],
        index=0
    )
    task_type = 'lung' if 'Lung' in task else 'colon'

    st.markdown("---")
    st.markdown("### 📊 Dataset Information")
    st.info("""
    **LC25000 Dataset**
    - 15,000 histopathology images
    - Resolution: 768×768 pixels
    - Architecture: ViT-B/16
    - Pre-trained: ImageNet-21k
    """)

      # Sample Images
    st.markdown("### 🖼️ Sample Images")
    st.caption("Click to load a sample image")
   
    sample_col1, sample_col2 = st.columns(2)
    with sample_col1:
        if st.button("Lung Benign"):
            st.session_state['use_sample'] = True
            st.session_state['sample_type'] = 'lung_benign'
    with sample_col2:
        if st.button("Lung ACA"):
            st.session_state['use_sample'] = True
            st.session_state['sample_type'] = 'lung_aca'
   
    st.markdown("---")
    st.markdown("### ⚙️ XAI Settings")
    show_attention = st.checkbox("Show Attention Rollout", value=True)
    show_gradcam   = st.checkbox("Show Grad-CAM", value=True)
    if show_attention:
        head_fusion   = st.selectbox("Attention Fusion:", ["mean", "min", "max"])
        discard_ratio = st.slider("Discard Ratio:", 0.0, 1.0, 0.9)
    else:
        head_fusion, discard_ratio = 'mean', 0.9

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("""
    **Authors:** Aya Gharsalli & Ferdaws Saidi

    **Supervisor:** Dr. Khemaies Abdallah

    **Course:** Deep Learning

    Built with PyTorch & Streamlit
    """)


# ============================================
# MAIN CONTENT
# ============================================
st.markdown('<h1 class="main-header">Histopathology Cancer Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Diagnosis with Explainable AI (XAI)</p>', unsafe_allow_html=True)

# Load model
model, device, class_names, model_info = load_model(task_type)

# Upload
st.markdown("### 📤 Upload Histopathology Image")
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Upload a histopathology image for analysis"
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.markdown("#### Supported Formats")
    st.markdown("- JPG / JPEG\n- PNG\n- TIFF\n\n**Recommended:** 768×768 resolution")
    st.markdown("---")
    st.markdown("#### Model Performance")
    for metric, value in model_info.items():
        st.metric(label=metric, value=value)

# ============================================
# PREDICTION
# ============================================
if uploaded_file:
    st.markdown("---")
    st.markdown("### 🔍 Analysis Results")

    with st.spinner("Analysing tissue sample..."):
        img_tensor = preprocess_image(image)

        with torch.no_grad():
            output     = model(img_tensor.unsqueeze(0).to(device))
            probs      = torch.softmax(output, dim=1)[0].cpu().numpy()
            pred_class = output.argmax(1).item()
            confidence = probs[pred_class]

        pred_col1, pred_col2 = st.columns([1, 1])

        with pred_col1:
            st.markdown("#### 🎯 Prediction")
            is_malignant = pred_class > 0
            box_class    = "malignant-box" if is_malignant else "benign-box"
            color        = "#dc2626" if is_malignant else "#16a34a"

            st.markdown(f"""
            <div class="prediction-box {box_class}">
                <h3 style="color:{color}; margin:0; font-size:1.8rem;">{class_names[pred_class]}</h3>
                <p style="color:{color}; margin:0.5rem 0 0 0; font-size:1.2rem;">Confidence: {confidence*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 📊 Confidence Scores")
            for i, (name, prob) in enumerate(zip(class_names, probs)):
                st.markdown(f"**{name}**: {prob*100:.2f}%")
                st.progress(float(prob))

        with pred_col2:
            fig = plot_confidence_scores(probs, class_names)
            st.pyplot(fig)
            plt.close()

    # ============================================
    # XAI
    # ============================================
    st.markdown("---")
    st.markdown("### 🧠 Explainable AI (XAI) Analysis")

    with st.spinner("Generating explanations..."):
        img_np = denormalize_image(img_tensor)

        attention_mask, grad_cam_mask = None, None

        if show_attention:
            attention_mask, _, _ = compute_attention_rollout(
                model, img_tensor, device,
                head_fusion=head_fusion,
                discard_ratio=discard_ratio
            )

        if show_gradcam:
            gradcam = ViTGradCAM(model)
            grad_cam_mask, _, _ = gradcam.generate(img_tensor, device)
            gradcam.remove_hooks()

        if attention_mask is not None and grad_cam_mask is not None:
            fig = create_xai_comparison(img_np, attention_mask, grad_cam_mask, class_names, pred_class)
            st.pyplot(fig)
            plt.close()
            st.markdown("""
            <div class="info-box">
                <h4>💡 Clinical Insight</h4>
                <p>The attention heatmap shows overall model focus, while Grad-CAM highlights regions
                specifically driving the prediction.</p>
                <p><strong>Note:</strong> This AI tool assists diagnosis but does not replace pathologist evaluation.</p>
            </div>
            """, unsafe_allow_html=True)

        elif attention_mask is not None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            attention_up = np.array(Image.fromarray((attention_mask * 255).astype(np.uint8)).resize((224, 224), Image.BILINEAR)) / 255.0
            ax[0].imshow(img_np); ax[0].set_title('Original'); ax[0].axis('off')
            ax[1].imshow(img_np); ax[1].imshow(attention_up, alpha=0.5, cmap='jet'); ax[1].set_title('Attention Rollout'); ax[1].axis('off')
            plt.tight_layout(); st.pyplot(fig); plt.close()

        elif grad_cam_mask is not None:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img_np); ax[0].set_title('Original'); ax[0].axis('off')
            ax[1].imshow(img_np); ax[1].imshow(grad_cam_mask, alpha=0.5, cmap='jet'); ax[1].set_title(f'Grad-CAM: {class_names[pred_class]}'); ax[1].axis('off')
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#94a3b8; padding:1rem 0;'>
    <p>CancerDetect AI • Built with PyTorch & Streamlit • Deep Learning Course Project</p>
    <p>Authors: Aya Gharsalli & Ferdaws Saidi • Supervisor: Dr. Khemaies Abdallah</p>
</div>
""", unsafe_allow_html=True)