import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# =======================
# background image
# =======================
def add_bg_from_local(image_file):
    """إضافة خلفية من صورة محلية"""
    with open(image_file, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# استدعاء الخلفية
add_bg_from_local("background.jpg")

# =======================
# عنوان التطبيق
# =======================
st.set_page_config(page_title="Brain Tumor Segmentation & Classification", page_icon="🧠", layout="wide")
st.title("🧠 Brain MRI Tumor Segmentation & Classification")
st.write("Upload an MRI image (and optional ground truth mask) to see segmentation and tumor type prediction.")

# =======================
# تحميل الموديلات
# =======================
@st.cache_resource
def load_models():
    if not os.path.exists("BrainTumorSegm.keras") or not os.path.exists("MRI_Class.keras"):
        st.error("❌ Model files not found! Make sure BrainTumorSegm.keras and MRI_Class.keras exist in the same folder as main.py.")
        st.stop()
    segmentation_model = load_model("BrainTumorSegm.keras")
    classification_model = load_model("MRI_Class.keras")
    return segmentation_model, classification_model

segmentation_model, classification_model = load_models()

# =======================
# دوال مساعدة
# =======================
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(128, 128), color_mode="grayscale")
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def adaptive_threshold_otsu(mask):
    mask = mask.squeeze().astype(np.float32)
    mask = (mask * 255).astype(np.uint8)
    _, binary = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# =======================
# واجهة Streamlit
# =======================
uploaded_mri = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])
uploaded_gt = st.file_uploader("Upload Ground Truth Mask (optional)", type=["jpg", "png", "jpeg"])

if uploaded_mri is not None:
    # Preprocess
    preprocessed_image = preprocess_image(uploaded_mri)

    # Segmentation Prediction
    raw_mask = segmentation_model.predict(preprocessed_image)
    predicted_mask = adaptive_threshold_otsu(raw_mask)
    predicted_mask = np.expand_dims(predicted_mask, axis=(0, -1))

    # Load Ground Truth
    gt_mask = None
    if uploaded_gt is not None:
        gt_img = image.load_img(uploaded_gt, target_size=(128, 128), color_mode="grayscale")
        gt_mask = image.img_to_array(gt_img)
        gt_mask = (gt_mask > 127).astype(np.uint8)

    # Metrics
    dice, iou = None, None
    if gt_mask is not None:
        dice = dice_coefficient(gt_mask, predicted_mask[0])
        iou = iou_score(gt_mask, predicted_mask[0])

    # Visualization
    fig, axes = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(15, 5))

    axes[0].imshow(preprocessed_image[0].squeeze(), cmap="gray")
    axes[0].set_title("MRI Image")
    axes[0].axis("off")

    if gt_mask is not None:
        axes[1].imshow(gt_mask.squeeze(), cmap="gray")
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis("off")
        mask_axis = axes[2]
    else:
        mask_axis = axes[1]

    mask_axis.imshow(preprocessed_image[0].squeeze(), cmap="gray")
    mask_axis.imshow(predicted_mask[0].squeeze(), cmap="Reds", alpha=0.5)
    if dice is not None and iou is not None:
        mask_axis.set_title(f"Predicted Mask\nDice: {dice:.4f}, IoU: {iou:.4f}")
    else:
        mask_axis.set_title("Predicted Mask")
    mask_axis.axis("off")

    st.pyplot(fig)

    # Classification Prediction
    tumor_type_prediction = classification_model.predict(preprocessed_image)
    class_labels = ["Glioma", "Meningioma", "No tumor", "Pituitary"]
    tumor_class = np.argmax(tumor_type_prediction, axis=1)[0]
    predicted_label = class_labels[tumor_class]

    st.subheader("🩺 Predicted Tumor Type:")
    st.success(predicted_label)
