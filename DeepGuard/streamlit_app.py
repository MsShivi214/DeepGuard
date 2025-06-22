import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.vgg16 import preprocess_input

# Load class names from labels.txt
LABELS_PATH = "converted_keras (1)/labels.txt"
with open(LABELS_PATH, "r") as f:
    lines = f.readlines()
    class_names = [line.strip().split(' ', 1)[1] for line in lines if line.strip()]

# Load the trained Keras model
MODEL_PATH = "deepguard_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (128, 128)

def preprocess_image(image: Image.Image):
    # Convert to RGB, resize, normalize
    img = image.convert('RGB')
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def predict_image(image: Image.Image):
    img = preprocess_image(image)
    pred = model.predict(img)[0][0]
    class_idx = int(pred > 0.5)
    class_name = class_names[class_idx]
    confidence = pred if pred > 0.5 else 1 - pred
    return class_name, float(confidence), float(pred)

# Streamlit UI
st.set_page_config(page_title="DeepGuard: Real vs Fake Image Detection", layout="centered")
st.title("🛡️ DeepGuard: Real vs Fake Image Detection")
st.write("Upload an image to detect if it is <span style='color:crimson'><b>Fake</b></span> or <span style='color:green'><b>Real</b></span>.", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    with st.spinner('Analyzing image...'):
        class_name, confidence, raw_score = predict_image(image)
    st.markdown(f"**Prediction:** <span style='color:blue;font-size:1.2em'>{class_name}</span>", unsafe_allow_html=True)
    st.markdown(f"**Confidence:** <span style='color:orange'>{confidence:.2%}</span>")
    st.markdown(f"**Raw Model Score:** {raw_score:.4f}")
    st.info(f"Debug: Model output (sigmoid): {raw_score:.4f} | Threshold: 0.5 | Mapping: 0=Fake, 1=Real")
    if (class_name == 'Real' and raw_score < 0.5) or (class_name == 'Fake' and raw_score > 0.5):
        st.warning("⚠️ The class mapping may be reversed. If predictions are consistently wrong, try swapping the class mapping in the code.")
else:
    st.info("Please upload an image to get started.")

st.markdown("---")
st.caption("DeepGuard | Powered by Keras & Streamlit") 