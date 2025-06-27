import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import base64

app = Flask(__name__)

# Load the trained model
model = load_model("Blood Cell.h5")
class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

def predict_image_class(image_path, model):
    # Read and preprocess the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_normalized = img_resized / 255.0
    img_preprocessed = img_normalized.reshape((1, 224, 224, 3))
    
    # Make prediction
    predictions = model.predict(img_preprocessed)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_label = class_labels[predicted_class_idx]
    
    # Get confidence scores
    confidence = {class_labels[i]: float(predictions[0][i]) for i in range(len(class_labels))}
    
    return predicted_class_label, img_rgb, confidence


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        # Check file extension
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return render_template("home.html", error="Invalid file type. Please upload a PNG, JPG, or JPEG image.")
        
        if file:
            try:
                # Verify the image can be read
                img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("Could not read image")
                
                # Reset file pointer
                file.seek(0)
                
                # Save uploaded file
                file_path = os.path.join("static", file.filename)
                file.save(file_path)
                
                # Make prediction
                predicted_class_label, img_rgb, confidence = predict_image_class(file_path, model)
                
                # Convert image to base64 for display
                _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                img_str = base64.b64encode(img_encoded).decode('utf-8')
                
                # Clean up uploaded file
                os.remove(file_path)
                
                return render_template("result.html", 
                                     class_label=predicted_class_label, 
                                     img_data=img_str,
                                     confidence=confidence)
            
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return render_template("home.html", error=f"Error processing image: {str(e)}")
    
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)  # Different port number


