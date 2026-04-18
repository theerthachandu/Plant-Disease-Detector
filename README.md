# 🌿 Plant Disease Detector

This project is a Deep Learning-based Streamlit web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN) built from scratch. It provides real-time predictions with confidence scores and a probability distribution graph across all 15 disease classes — making plant disease diagnosis accessible to anyone without technical knowledge.

---

## 🔍 Features

- Upload a leaf image and get instant prediction  
- Detects plant leaf diseases across **15 different classes**  
- Displays:  
  - Predicted class  
  - Confidence score  
  - Class-wise probability distribution (graph)  
  - Clear **HEALTHY ✅ or DISEASED ⚠️** verdict  
- Intuitive and lightweight web interface using Streamlit  
- Model hosted externally via Google Drive (to keep repo clean)

---

## 🛠️ Technologies & Libraries Used

- **Python 3.10+**
- **TensorFlow / Keras** – CNN model building, training and inference
- **Streamlit** – Interactive web application interface and deployment
- **gdown** – Downloads trained `.keras` model from Google Drive at runtime
- **NumPy** – Array operations and Softmax output processing
- **Matplotlib** – Plotting class-wise probability distribution graph
- **PIL (Pillow)** – Image loading, resizing and preprocessing
- **Scikit-learn** – Confusion matrix and classification report generation
- **Seaborn** – Confusion matrix heatmap visualization
- **ImageDataGenerator (Keras)** – Data augmentation during training
- **Google Colab** – Cloud GPU environment used for model training

---

## 📈 Model Performance

- Trained using Convolutional Neural Networks (CNN)  
- Dataset: **PlantVillage** with ~20,638 images  
- **Train / Val Split**: 80:20 — 16,510 train / 4,128 validation  
- **Training Accuracy**: ~87%  
- **Validation Accuracy**: ~84%  
- Evaluated using classification report, confusion matrix, and probability graphs  
- Class imbalance handled with data augmentation (flips, zoom, rotation) and callbacks  

> 🔎 Note: While the model performs well, minor misclassifications may occur due to overlapping symptoms between visually similar classes such as Tomato Target Spot, Late Blight and Early Blight — a known challenge even for human experts in plant pathology.

---

## 🌐 Live Demo

Click below to try the working deployed version:  
🔗 **[Plant Disease Detector Web App](https://plant-disease-detector-tyyikxu4je3gdxxehkogfb.streamlit.app/)**

---

## 📦 Files Included

- `app.py` – Streamlit app file  
- `requirements.txt` – Required Python libraries  
- `plant_leaf_dis.keras` – Trained CNN model (hosted externally via Google Drive)

---

## 📁 Dataset

- Source: **PlantVillage Dataset** (Kaggle)
- Total Images: ~20,638
- Number of Classes: **15**
- Train / Validation Split: **80:20**
- Categories include:

  **Tomato (10 classes)** — Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy

  **Potato (3 classes)** — Early Blight, Late Blight, Healthy

  **Pepper Bell (2 classes)** — Bacterial Spot, Healthy

---




