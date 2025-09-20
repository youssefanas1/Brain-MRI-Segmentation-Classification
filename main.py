import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import base64
import google.generativeai as genai


st.set_page_config(page_title="Brain MRI Classifier", layout="wide")

# ----------------------------
# Background + card
# ----------------------------
def add_bg_and_card(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        min-height: 100vh;
    }}

    .block-container {{
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        padding: 30px 40px;
        margin-top: 40px;
        margin-bottom: 40px;
        max-width: 950px;
        margin-left: auto;
        margin-right: auto;
        color: black !important;
    }}

    /* Uploader box */
    .stFileUploader > div > div {{
        border: 2px dashed #3399ff !important;
        background: rgba(0,0,0,0.05) !important;
        border-radius: 12px !important;
    }}

    /* Browse button */
    .stFileUploader button {{
        background-color: #3399ff !important;
        color: white !important;
        border-radius: 6px !important;
        padding: 0.4rem 1rem !important;
        border: none !important;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

add_bg_and_card("background.jpeg")

# ----------------------------
# Config
# ----------------------------
IMG_SIZE = 128
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

@st.cache_resource
def load_models():
    classifier = load_model("MRI_Class.keras")
    segmenter = load_model("SegmentMRI.keras")
    return classifier, segmenter

classifier, segmenter = load_models()

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_image(img: Image.Image):
    img = img.convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    arr = np.expand_dims(arr, axis=0)
    return arr

# ----------------------------
# Overlay mask (resize image down to mask size)
# ----------------------------
def overlay_mask(image_pil, mask, threshold=0.6, alpha=0.4):
    mask = np.squeeze(mask)

    # Convert mask → binary image
    mask_img = Image.fromarray((mask > threshold).astype(np.uint8) * 255)

    # Resize MRI down to mask size
    image_resized = image_pil.resize(mask_img.size)

    # Build RGBA overlay
    mask_arr = np.array(mask_img)
    overlay = np.zeros((mask_img.size[1], mask_img.size[0], 4), dtype=np.uint8)
    overlay[:, :, 0] = mask_arr
    overlay[:, :, 3] = (mask_arr > 0) * int(255 * alpha)

    combined = Image.alpha_composite(image_resized.convert("RGBA"), Image.fromarray(overlay))

    return combined  # stays same size as mask, no scaling back up

# ----------------------------
# Gemini Setup
# ----------------------------
# ----------------------------
# Gemini Setup (Fixed)
# ----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])


# create a single model object once
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def generate_report(tumor_type, confidence):
    prompt = f"""
    You are a medical assistant. Based on the MRI classifier result:
    - Tumor type: {tumor_type}
    - Confidence: {confidence:.2f}

    Please provide a short report including:
    1. A description of the tumor type
    2. Common symptoms
    3. Typical treatment options
    4. A disclaimer that this is AI-generated and not a medical diagnosis.

    If the tumor type is "No Tumor", just return that the patient is healthy and no report.
    """
    response = gemini_model.generate_content(prompt)
    return response.text


# ----------------------------
# UI
# ----------------------------
st.title("🧠 Brain MRI Classification & Segmentation")
st.write("Upload a brain MRI scan to classify tumor type and segment tumor region.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    preprocessed = preprocess_image(image)

    raw_pred = segmenter.predict(preprocessed)[0]
    overlay_img = overlay_mask(image, raw_pred, threshold=0.47, alpha=0.5)

    st.subheader("Segmentation Result")
    st.image(overlay_img, caption="Overlay (resized to mask)", use_container_width=True)

    class_probs = classifier.predict(preprocessed)[0]
    pred_idx = np.argmax(class_probs)
    pred_class = CLASSES[pred_idx]
    confidence = float(class_probs[pred_idx])

    st.markdown(f"<h2 style='color:black;'>Prediction: <b>{pred_class}</b></h2>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='color:black;'>Confidence: {confidence:.2f}</h4>", unsafe_allow_html=True)

    fig, ax = plt.subplots()
    ax.bar(CLASSES, class_probs, color="skyblue")
    ax.set_ylabel("Probability", color="black")
    ax.set_ylim([0, 1])
    ax.tick_params(colors="black")
    for i, v in enumerate(class_probs):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center", color="black")
    st.pyplot(fig)

    # ----------------------------
    # Gemini Report
    # ----------------------------
    st.subheader("📋 AI-Generated Medical Report")
    with st.spinner("Generating report..."):
        report = generate_report(pred_class, confidence)
    st.write(report)