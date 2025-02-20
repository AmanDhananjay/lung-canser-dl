from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load Trained Model
model = tf.keras.models.load_model("models/lung_cancer_model.h5")
IMG_SIZE = 128

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    prediction = model.predict(img)[0][0]
    result = "Cancerous" if prediction > 0.5 else "Normal"
    
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
