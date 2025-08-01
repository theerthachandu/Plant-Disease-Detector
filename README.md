# 🌿 Plant Disease Detector

This project is a Deep Learning-based Streamlit web application that detects plant diseases from leaf images using a Convolutional Neural Network (CNN). It provides real-time predictions with confidence scores and a probability distribution graph for all disease classes.

---

## 🔍 Features

- Upload a leaf image and get instant prediction  
- Detects plant leaf diseases across **15 different classes**  
- Displays:  
  - Predicted class  
  - Confidence score  
  - Class-wise probability distribution (graph)  
- Intuitive and lightweight web interface using Streamlit  
- Model hosted externally via Google Drive (to keep repo clean)

---

## 🛠️ Technologies & Libraries Used

- **Python 3.10+**
- **TensorFlow / Keras** – Model training and prediction  
- **Streamlit** – Interactive web app interface  
- **gdown** – Download model (`.keras`) from Google Drive  
- **NumPy** – Numerical computation  
- **Matplotlib** – Plotting class probabilities  
- **PIL (Pillow)** – Image preprocessing and resizing

---

## 📈 Model Performance

- Trained using Convolutional Neural Networks (CNN)  
- Dataset: **PlantVillage** with ~20,638 images  
- **Training Accuracy**: ~87%  
- **Validation Accuracy**: ~84%  
- Evaluated using classification report, confusion matrix, and probability graphs  
- Some class imbalance handled with data augmentation and callbacks  

> 🔎 Note: While the model performs well, minor misclassifications may occur due to overlapping symptoms or imbalanced class distribution.

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
- Categories include:  
  - Tomato Yellow Leaf Curl Virus  
  - Potato Early Blight  
  - Pepper Bacterial Spot  
  - And 12 more diseases/conditions

---




