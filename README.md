# 🩸 HematoVision - Blood Cell Classification using Deep Learning

HematoVision is an AI-powered web application that classifies different types of blood cells using deep learning (MobileNetV2). The app allows users to upload blood cell images and get instant predictions, assisting in the diagnosis of blood disorders.

## 🔧 Features
- Classifies 4 types of white blood cells: Eosinophils, Lymphocytes, Monocytes, Neutrophils
- Built with MobileNetV2 and Flask
- Interactive web interface for file upload
- Confusion matrix and evaluation metrics included

## 🗂️ Folder Structure
- `Backend Code/`: Contains Flask API and server logic
- `Frontend Pages/`: HTML templates (home & result page)
- `Model Training/`: Model building, training notebook and saved `.h5` model
- `Documentation/`: Report, video demonstration, and supporting docs

## ✅ Requirements
- Python 3.8+
- TensorFlow, Flask, OpenCV, NumPy, Matplotlib

## 🚀 Run the App
```bash
cd Backend\ Code
python app.py
