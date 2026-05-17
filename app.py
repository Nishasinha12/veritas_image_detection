"""
Veritas — Image Deepfake Detection Microservice
================================================
Standalone Flask service for image-based deepfake detection.
"""

import os
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_image_model.h5")

# Global model reference
model = None

# ---------------------------------------------------------------------------
# Download model from Hugging Face
# ---------------------------------------------------------------------------
def download_models():
    """
    Downloads the image model from Hugging Face if not already present.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        print("⬇️ Downloading deepfake_image_model.h5 from Hugging Face...")

        try:
            hf_hub_download(
                repo_id="nishuu12/veritas-models",
                filename="deepfake_image_model.h5",
                local_dir=MODEL_DIR,
                token=os.getenv("HF_TOKEN")
            )

            print("✅ Image model downloaded successfully")

        except Exception as e:
            print(f"⚠️ Hugging Face download failed: {e}")
            traceback.print_exc()

    else:
        print("✅ Image model already exists locally")


# ---------------------------------------------------------------------------
# Lazy-load TensorFlow model
# ---------------------------------------------------------------------------
def get_model():
    """
    Loads TensorFlow model only when first prediction request comes.
    Prevents Cloud Run startup crashes.
    """

    global model

    if model is None:

        # Download model if needed
        download_models()

        if os.path.isfile(MODEL_PATH):

            try:
                import tensorflow as tf

                print(f"🔄 Loading TensorFlow model from {MODEL_PATH}...")

                model = tf.keras.models.load_model(MODEL_PATH)

                print("✅ Image model loaded successfully!")

            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                traceback.print_exc()
                model = None

        else:
            print("ℹ️ Model file missing — using stub mode.")

    return model


# ---------------------------------------------------------------------------
# Stub response if model unavailable
# ---------------------------------------------------------------------------
def _stub_prediction():
    return {
        "prediction": "Deepfake",
        "confidence": 0.85,
        "note": "model pending"
    }


# ---------------------------------------------------------------------------
# Real TensorFlow inference
# ---------------------------------------------------------------------------
def _run_inference(loaded_model, img_array: np.ndarray) -> dict:

    prediction = loaded_model.predict(img_array)

    raw_score = float(prediction[0][0])

    if raw_score > 0.5:
        label = "Real"
        confidence = raw_score
    else:
        label = "Deepfake"
        confidence = 1.0 - raw_score

    return {
        "prediction": label,
        "confidence": round(confidence, 4)
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "running",
        "service": "Veritas Image API"
    })


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "ok",
        "service": "image",
        "model_loaded": model is not None
    })


@app.route("/predict/image", methods=["POST"])
def predict_image():

    try:

        # -------------------------------------------------------------------
        # Validate upload
        # -------------------------------------------------------------------
        if "image" not in request.files:
            return jsonify({
                "error": "No image file provided. Use form-field name 'image'."
            }), 400

        file = request.files["image"]

        if file.filename == "":
            return jsonify({
                "error": "Empty filename."
            }), 400

        # -------------------------------------------------------------------
        # Open image safely
        # -------------------------------------------------------------------
        try:
            img = Image.open(file.stream).convert("RGB")

        except Exception:
            return jsonify({
                "error": "Could not open file as an image."
            }), 400

        # -------------------------------------------------------------------
        # Preprocessing
        # -------------------------------------------------------------------
        img = img.resize((256, 256))

        img_array = np.array(img, dtype=np.float32) / 255.0

        img_array = np.expand_dims(img_array, axis=0)

        # -------------------------------------------------------------------
        # Load model lazily
        # -------------------------------------------------------------------
        active_model = get_model()

        # -------------------------------------------------------------------
        # Predict
        # -------------------------------------------------------------------
        if active_model is not None:

            result = _run_inference(active_model, img_array)

        else:

            result = _stub_prediction()

        return jsonify(result), 200

    except Exception as e:

        print(f"❌ /predict/image error: {e}")

        traceback.print_exc()

        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5002))

    print(f"🚀 Image service starting on 0.0.0.0:{port}")

    app.run(host="0.0.0.0", port=port)