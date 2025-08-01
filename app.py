import streamlit as st 
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os
import matplotlib.pyplot as plt

# --------------------------
# Page Setup
# --------------------------
st.set_page_config(page_title="\U0001F33F Plant Disease Detector", layout="centered")
st.title("\U0001F33F Plant Disease Detector")  # leaf emoji code

# --------------------------
# Load Model (from Google Drive via gdown)
# --------------------------
@st.cache_resource
def load_model():
    file_id = "1cXjIkuMIHNtQ4rvo9fjsKRG0utsB4CQ5"  # Your file ID
    model_path = "plant_leaf_dis.keras"
    if not os.path.exists(model_path):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()

# --------------------------
# Class Labels
# --------------------------
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
               'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
               'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
               'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite',
               'Tomato__Target_Spot', 'Tomato__Tomato_YellowLeaf__Curl_Virus',
               'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# --------------------------
# Upload & Predict
# --------------------------
uploaded_file = st.file_uploader("\U0001F4F8 Upload a leaf image...", type=["jpg", "jpeg", "png"])  # cam emoji

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="\U0001F5BC Uploaded Image", use_container_width=True)  # pic emoji

    st.write("\U0001F50D Classifying...")  # magnifying glass emoji

    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    # --------------------------
    # Probability Graph
    # --------------------------
    st.subheader("\U0001F4CA Class Probability Distribution")  # graph emoji
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(class_names, predictions[0], color='skyblue')
    ax.set_xlabel('Confidence')
    ax.set_title('Prediction Confidence per Class')
    ax.invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center')
    plt.tight_layout()
    st.pyplot(fig)

    # --------------------------
    # Output Message
    # --------------------------
    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")

    if "healthy" in predicted_class.lower():
        st.success("\u2705 The plant is HEALTHY")  # correct emoji
    else:
        st.warning(f"\u26A0\uFE0F The plant is DISEASED with: **{predicted_class}**")  # danger signal emoji
