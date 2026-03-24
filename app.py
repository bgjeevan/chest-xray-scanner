import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os
import gdown
from PIL import Image

st.set_page_config(
    page_title="RadScan AI — Chest X-Ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background-color: #0F1117;
        color: #E8EAF0;
    }

    .main-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
        border-bottom: 1px solid #2a3040;
        padding: 2rem 3rem;
        margin: -1rem -1rem 2rem -1rem;
    }

    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
        color: #60a5fa;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #f0f4ff;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .header-subtitle {
        color: #8892a4;
        font-size: 0.95rem;
        margin-top: 6px;
        font-weight: 400;
    }

    .upload-zone {
        background: #1a1f2e;
        border: 2px dashed #2a3a5c;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 1.5rem;
    }

    .upload-zone:hover {
        border-color: #3b82f6;
        background: #1e2638;
    }

    .result-card {
        background: #1a1f2e;
        border: 1px solid #2a3040;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #c8d0e0;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2a3040;
    }

    .finding-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 0;
        border-bottom: 1px solid #1e2535;
    }

    .finding-row:last-child { border-bottom: none; }

    .finding-name {
        font-size: 0.9rem;
        color: #c8d0e0;
        font-weight: 500;
    }

    .badge-detected {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-normal {
        background: rgba(34, 197, 94, 0.1);
        color: #4ade80;
        border: 1px solid rgba(34, 197, 94, 0.2);
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .score-text {
        font-size: 0.85rem;
        color: #8892a4;
        min-width: 45px;
        text-align: right;
    }

    .top-finding-card {
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.2);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .top-finding-label {
        font-size: 0.75rem;
        color: #f87171;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .top-finding-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #fff;
    }

    .confidence-chip {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .disclaimer-box {
        background: rgba(234, 179, 8, 0.08);
        border: 1px solid rgba(234, 179, 8, 0.2);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-top: 1.5rem;
    }

    .disclaimer-text {
        color: #fbbf24;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0;
    }

    .img-label {
        font-size: 0.8rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #60a5fa);
        border-radius: 10px;
    }

    .stProgress > div > div {
        background: #1e2535;
        border-radius: 10px;
    }

    div[data-testid="stFileUploader"] {
        background: #1a1f2e;
        border: 2px dashed #2a3a5c;
        border-radius: 16px;
        padding: 1rem;
    }

    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }

    footer { display: none; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural Thickening', 'Hernia'
]

MODEL_PATH = 'pretrained_model.h5'
GDRIVE_ID = '1qTEW8ftNXU7wIHaGIZVIrIYEWEaGwkGQ'

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading AI model..."):
            gdown.download(
                f'https://drive.google.com/uc?id={GDRIVE_ID}',
                MODEL_PATH, quiet=False
            )
    base_model = DenseNet121(include_top=False, weights=None, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(14, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights(MODEL_PATH)
    return model

def generate_heatmap(model, img_array, pred_index):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer = layer
            break
    grad_model = Model(inputs=model.inputs, outputs=[last_conv_layer.output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(np.array(heatmap), 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return heatmap

def overlay_heatmap(img_path, heatmap):
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))
    heatmap_resized = cv2.resize(np.array(heatmap), (224, 224))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original_img, 0.55, heatmap_colored, 0.45, 0)
    return cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)

# Header
st.markdown("""
<div class="main-header">
    <div class="header-badge">AI-Powered · DenseNet121 · 14 Conditions</div>
    <h1 class="header-title">🫁 RadScan AI</h1>
    <p class="header-subtitle">Upload a chest X-ray for instant AI-assisted analysis across 14 pulmonary conditions</p>
</div>
""", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray (JPG, JPEG, PNG)",
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    temp_path = "temp_xray.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Running AI analysis..."):
        model = load_model()
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = np.expand_dims(image.img_to_array(img), axis=0) / 255.0
        preds = model.predict(img_array)[0]
        top_index = np.argmax(preds)
        heatmap = generate_heatmap(model, img_array, top_index)
        result_img = overlay_heatmap(temp_path, heatmap)

    # Top finding banner
    st.markdown(f"""
    <div class="top-finding-card">
        <div style="flex:1">
            <div class="top-finding-label">Primary Finding</div>
            <div class="top-finding-value">{LABELS[top_index]}</div>
        </div>
        <div class="confidence-chip">{preds[top_index]*100:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

    # Images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="img-label">Original X-Ray</div>', unsafe_allow_html=True)
        st.image(uploaded_file, use_column_width=True)
    with col2:
        st.markdown(f'<div class="img-label">AI Heatmap — {LABELS[top_index]}</div>', unsafe_allow_html=True)
        st.image(result_img, use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Results table
    st.markdown('<div class="section-title">Full Scan Results</div>', unsafe_allow_html=True)

    detected = [(l, s) for l, s in zip(LABELS, preds) if s > 0.5]
    not_detected = [(l, s) for l, s in zip(LABELS, preds) if s <= 0.5]

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**Detected ({len(detected)})**")
        for label, score in sorted(detected, key=lambda x: x[1], reverse=True):
            c1, c2, c3 = st.columns([3, 4, 1])
            c1.markdown(f'<span class="finding-name">{label}</span>', unsafe_allow_html=True)
            c2.progress(float(score))
            c3.markdown(f'<span class="score-text">{score*100:.0f}%</span>', unsafe_allow_html=True)

    with col_b:
        st.markdown(f"**Not Detected ({len(not_detected)})**")
        for label, score in sorted(not_detected, key=lambda x: x[1], reverse=True):
            c1, c2, c3 = st.columns([3, 4, 1])
            c1.markdown(f'<span style="color:#4ade80;font-size:0.9rem">{label}</span>', unsafe_allow_html=True)
            c2.progress(float(score))
            c3.markdown(f'<span class="score-text">{score*100:.0f}%</span>', unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <p class="disclaimer-text">⚠️ For clinical assistance only. This AI analysis must be reviewed and confirmed by a qualified radiologist before any medical decision is made.</p>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #4a5568;">
        <div style="font-size: 4rem; margin-bottom: 1rem;">🫁</div>
        <div style="font-size: 1.1rem; color: #8892a4; font-weight: 500;">Upload a chest X-ray to begin analysis</div>
        <div style="font-size: 0.85rem; color: #4a5568; margin-top: 8px;">Supports JPG, JPEG, PNG formats</div>
    </div>
    """, unsafe_allow_html=True)
