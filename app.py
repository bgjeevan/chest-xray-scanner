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
    .stApp { background-color: #0F1117; color: #E8EAF0; }
    footer { display: none; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }

    .main-header {
        background: #13161f;
        border-bottom: 1px solid #1e2535;
        padding: 1.8rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
    }
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(59, 130, 246, 0.12);
        border: 1px solid rgba(59, 130, 246, 0.25);
        color: #60a5fa;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 700;
        color: #f0f4ff;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        color: #6b7a99;
        font-size: 0.9rem;
        margin-top: 5px;
    }

    .primary-finding {
        background: #13161f;
        border: 1px solid #1e2535;
        border-left: 3px solid #ef4444;
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .pf-label {
        font-size: 10px;
        color: #ef4444;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .pf-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #fff;
    }
    .pf-badge {
        background: rgba(239,68,68,0.12);
        color: #f87171;
        border: 1px solid rgba(239,68,68,0.25);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 700;
    }

    .img-container {
        background: #13161f;
        border: 1px solid #1e2535;
        border-radius: 12px;
        overflow: hidden;
        margin-bottom: 1.5rem;
    }
    .img-header {
        padding: 10px 14px;
        border-bottom: 1px solid #1e2535;
        font-size: 10px;
        font-weight: 700;
        color: #6b7a99;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }
    .img-body { padding: 12px; }

    .results-header {
        font-size: 10px;
        font-weight: 700;
        color: #6b7a99;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 1rem;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e2535;
    }

    .finding-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0;
        border-bottom: 1px solid #13161f;
    }
    .finding-item:last-child { border-bottom: none; }
    .finding-dot-red {
        width: 7px; height: 7px;
        background: #ef4444;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .finding-dot-green {
        width: 7px; height: 7px;
        background: #22c55e;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .finding-label {
        font-size: 0.85rem;
        color: #c8d0e0;
        font-weight: 500;
        min-width: 130px;
    }
    .bar-track {
        flex: 1;
        background: #1a1f2e;
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
    }
    .bar-fill-red {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #dc2626, #f87171);
    }
    .bar-fill-green {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #15803d, #4ade80);
    }
    .finding-score {
        font-size: 0.8rem;
        font-weight: 600;
        min-width: 38px;
        text-align: right;
    }
    .score-red { color: #f87171; }
    .score-green { color: #4ade80; }

    .section-card {
        background: #13161f;
        border: 1px solid #1e2535;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    .disclaimer-box {
        background: rgba(234,179,8,0.06);
        border: 1px solid rgba(234,179,8,0.18);
        border-radius: 10px;
        padding: 0.9rem 1.2rem;
        margin-top: 1.5rem;
    }
    .disclaimer-text {
        color: #fbbf24;
        font-size: 0.8rem;
        font-weight: 500;
        margin: 0;
    }

    .empty-state {
        text-align: center;
        padding: 5rem 2rem;
        color: #3a4258;
    }
    .empty-icon { font-size: 3.5rem; margin-bottom: 1rem; }
    .empty-title { font-size: 1rem; color: #6b7a99; font-weight: 500; }
    .empty-sub { font-size: 0.82rem; color: #3a4258; margin-top: 6px; }
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
            gdown.download(f'https://drive.google.com/uc?id={GDRIVE_ID}', MODEL_PATH, quiet=False)
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

# ── Header ──
st.markdown("""
<div class="main-header">
    <div class="header-badge">AI-Powered · DenseNet121 · 14 Conditions</div>
    <h1 class="header-title">🫁 RadScan AI</h1>
    <p class="header-subtitle">Upload a chest X-ray for instant AI-assisted analysis</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Chest X-Ray (JPG, JPEG, PNG)", type=['jpg', 'jpeg', 'png'])

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

    # Primary finding
    st.markdown(f"""
    <div class="primary-finding">
        <div>
            <div class="pf-label">Primary Finding</div>
            <div class="pf-value">{LABELS[top_index]}</div>
        </div>
        <div class="pf-badge">{preds[top_index]*100:.1f}% confidence</div>
    </div>
    """, unsafe_allow_html=True)

    # Images — smaller and contained
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="img-container"><div class="img-header">Original X-Ray</div><div class="img-body">', unsafe_allow_html=True)
        st.image(uploaded_file, width=320)
        st.markdown('</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="img-container"><div class="img-header">AI Heatmap — {LABELS[top_index]}</div><div class="img-body">', unsafe_allow_html=True)
        st.image(result_img, width=320)
        st.markdown('</div></div>', unsafe_allow_html=True)

    # Results
    detected = sorted([(l, s) for l, s in zip(LABELS, preds) if s > 0.5], key=lambda x: x[1], reverse=True)
    not_detected = sorted([(l, s) for l, s in zip(LABELS, preds) if s <= 0.5], key=lambda x: x[1], reverse=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f'<div class="section-card"><div class="results-header">Detected ({len(detected)})</div>', unsafe_allow_html=True)
        for label, score in detected:
            pct = int(score * 100)
            st.markdown(f"""
            <div class="finding-item">
                <div class="finding-dot-red"></div>
                <div class="finding-label">{label}</div>
                <div class="bar-track"><div class="bar-fill-red" style="width:{pct}%"></div></div>
                <div class="finding-score score-red">{pct}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown(f'<div class="section-card"><div class="results-header">Not Detected ({len(not_detected)})</div>', unsafe_allow_html=True)
        for label, score in not_detected:
            pct = int(score * 100)
            st.markdown(f"""
            <div class="finding-item">
                <div class="finding-dot-green"></div>
                <div class="finding-label">{label}</div>
                <div class="bar-track"><div class="bar-fill-green" style="width:{pct}%"></div></div>
                <div class="finding-score score-green">{pct}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        <p class="disclaimer-text">⚠️ For clinical assistance only. This AI analysis must be reviewed and confirmed by a qualified radiologist before any medical decision is made.</p>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists(temp_path):
        os.remove(temp_path)

else:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🫁</div>
        <div class="empty-title">Upload a chest X-ray to begin analysis</div>
        <div class="empty-sub">Supports JPG, JPEG, PNG formats</div>
    </div>
    """, unsafe_allow_html=True)
