
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import gdown
import os

st.set_page_config(page_title="🌿 Plant Disease Detector", layout="centered")
st.title("🌿 Plant Disease Detector")

# ⬇️ Model Download from Google Drive
@st.cache_resource
def load_model():
    file_id = '1Kt4J83VidPp2Em0YgyA6QdUCF6aVZBAY'
    model_path = 'plant_disease_model.keras'
    if not os.path.exists(model_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

model = load_model()

# 🏷️ Classes the model can predict
class_names = ['Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight',
               'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
               'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
               'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy']

# 📤 File uploader
uploaded_file = st.file_uploader("📸 Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='🖼️ Uploaded Image', use_column_width=True)
    st.write("🔍 Classifying...")
    
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    if "healthy" in predicted_class.lower():
        status = "✅ The plant is HEALTHY"
    else:
        status = f"⚠️ The plant is DISEASED with: **{predicted_class}**"

    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}")
    st.markdown(f"### {status}")
