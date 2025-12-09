import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# -----------------------------
# LOAD MODEL & LABELS
# -----------------------------
model = tf.keras.models.load_model("model.h5")

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

IMG_SIZE = (224, 224)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# -----------------------------
# API ROUTES
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "Please upload an image"}), 400

        image_file = request.files["image"]
        img = preprocess_image(image_file.read())

        predictions = model.predict(img)
        class_index = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))

        return jsonify({
            "prediction": labels[class_index],
            "confidence": confidence
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
