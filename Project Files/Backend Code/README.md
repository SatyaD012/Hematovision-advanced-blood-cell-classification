# 🧠 Backend - Flask Application

This folder contains the Flask backend for the HematoVision app.

## File: `app.py`
- Loads the trained model (`BloodCell.h5`)
- Accepts image file uploads from users
- Processes and classifies the image using MobileNetV2
- Returns the predicted class and displays it along with the image

## Libraries Used:
- Flask
- TensorFlow
- OpenCV
- NumPy
- Base64

## Run the App
```bash
python app.py
