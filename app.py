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
    page_title="Chest X-Ray AI Scanner",
    page_icon="🫁",
    layout="centered"
)

LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax',
    'Consolidation', 'Edema', 'Emphysema', 'Fibrosis',
    'Pleural Thickening', 'Hernia'
]

MODEL_PATH = 'pretrained_model.h5'

@st.cache_resource
def load_model():
    base_model = DenseNet121(
        include_top=False,
        weights=None,
        input_shape=(224, 224, 3)
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
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
    grad_model = Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = np.maximum(np.array(heatmap), 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    return heatmap

def overlay_heatmap(img_path, heatmap):
    original_img = cv2.imread(img_path)
    original_img = cv2.resize(original_img, (224, 224))
    heatmap_resized = cv2.resize(np.array(heatmap), (224, 224))
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(original_img, 0.6, heatmap_colored, 0.4, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    return superimposed_rgb

# Header
st.title("🫁 Chest X-Ray AI Scanner")
st.markdown("Upload a chest X-ray and the AI will scan it for 14 different conditions instantly.")
st.markdown("---")

# Upload
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray",
    type=['jpg', 'jpeg', 'png'],
    help="Upload any chest X-ray image"
)

if uploaded_file is not None:
    temp_path = "temp_xray.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original X-Ray")
        st.image(uploaded_file, use_column_width=True)

    with st.spinner("🔍 Scanning X-ray... please wait"):
        model = load_model()
        img = image.load_img(temp_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        preds = model.predict(img_array)[0]
        top_pred_index = np.argmax(preds)
        heatmap = generate_heatmap(model, img_array, top_pred_index)
        result_img = overlay_heatmap(temp_path, heatmap)

    with col2:
        st.subheader("AI Heatmap")
        st.image(result_img, use_column_width=True)
        st.caption(f"Highlighting: {LABELS[top_pred_index]}")

    st.markdown("---")
    st.subheader("📊 Full Scan Results")
    st.markdown("🔴 Detected   🟢 Not detected")
    st.markdown("")

    for label, score in zip(LABELS, preds):
        col_a, col_b, col_c = st.columns([2, 3, 1])
        with col_a:
            indicator = "🔴" if score > 0.5 else "🟢"
            st.write(f"{indicator} {label}")
        with col_b:
            st.progress(float(score))
        with col_c:
            st.write(f"{score*100:.1f}%")

    st.markdown("---")
    st.warning("⚠️ This AI tool is for assistance only. Always consult a qualified radiologist for final diagnosis.")

    if os.path.exists(temp_path):
        os.remove(temp_path)
