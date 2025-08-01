# 🌿 Plant Disease Detector

This project is a Deep Learning-based Streamlit web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN). It provides real-time predictions with confidence scores and a probability distribution graph for all disease classes.

---

## 🔍 Features

- Upload a leaf image and get instant prediction
- Detects plant leaf diseases across **15 different classes**
- Displays:
  - Predicted class
  - Confidence score
  - Full class probability distribution (graph)
- Simple and intuitive web interface using Streamlit
- Lightweight app with model hosted externally (via Google Drive)

---

## 🛠️ Technologies & Libraries Used

- **Python**
- **TensorFlow / Keras** – Model building and training
- **Streamlit** – Web app framework
- **gdown** – To download `.keras` model from Google Drive
- **NumPy** – Numerical operations
- **Matplotlib** – Class probability plots
- **PIL (Pillow)** – Image loading and resizing

---

## 📦 Files Included

- `app.py` – Streamlit application script
- `requirements.txt` – Required Python libraries
- `plant_leaf_dis.keras` – Trained model file (hosted on Google Drive, not in repo)

---

## 📁 Dataset

- Dataset used: **PlantVillage**  
- Total images: ~20,638  
- Classes: 15 plant conditions (e.g., Tomato Yellow Leaf Curl Virus, Potato Early Blight, Pepper Bacterial Spot, etc.)

---

