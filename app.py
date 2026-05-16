"""
Veritas — Image Deepfake Detection Microservice
================================================
Standalone Flask service for image-based deepfake detection.

• If a trained model exists at models/deepfake_image_model.h5, loads it and
  runs real inference on uploaded images (resized to 256×256).
• If NO model is found, returns a deterministic stub response so the service
  can still deploy and the frontend does not break.

Endpoints
---------
POST /predict/image   — accepts multipart/form-data with an image file
GET  /health          — lightweight health-check
"""

import os
import sys
import traceback
import gdown  # ← add this import

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Model loading (graceful fallback to stub)
# ---------------------------------------------------------------------------
# ─── STEP 1: Download model FIRST ───
def download_models():
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "deepfake_image_model.h5")
    if not os.path.exists(model_path):
        print("⬇️ Downloading deepfake_image_model.h5 from Google Drive...")
        gdown.download(
            "https://drive.google.com/uc?id=1cVahwkpTw_Fl3QEadgx_42GBcNoTE7dn",
            model_path,
            quiet=False
        )
        print("✅ Image model downloaded successfully")
    else:
        print("✅ Image model already exists locally")

download_models()  # ← call before model loading

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_image_model.h5")

model = None  # Will stay None when no model file is present

if os.path.isfile(MODEL_PATH):
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Image model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  Failed to load model at {MODEL_PATH}: {e}")
        traceback.print_exc()
        model = None
else:
    print(f"ℹ️  No model file found at {MODEL_PATH} — running in STUB mode.")


def _stub_prediction():
    """Return a fixed response when no model is available."""
    return {
        "prediction": "Deepfake",
        "confidence": 0.85,
        "note": "model pending",
    }


def _run_inference(img_array: np.ndarray) -> dict:
    """Run the real TF model and interpret the output."""
    prediction = model.predict(img_array)
    raw_score = float(prediction[0][0])

    if raw_score > 0.5:
        label = "Real"
        confidence = raw_score
    else:
        label = "Deepfake"
        confidence = 1.0 - raw_score

    return {
        "prediction": label,
        "confidence": round(confidence, 4),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    """Lightweight health-check endpoint."""
    try:
        return jsonify({
            "status": "ok",
            "service": "image",
            "model_loaded": model is not None,
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "detail": str(e)}), 500


@app.route("/predict/image", methods=["POST"])
def predict_image():
    """
    Accept an image via multipart/form-data, resize to 256×256,
    and return a deepfake prediction.

    Form field: ``image`` (file)
    """
    try:
        # --- Validate upload ------------------------------------------------
        if "image" not in request.files:
            return jsonify({"error": "No image file provided. Use form-field name 'image'."}), 400

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "Empty filename."}), 400

        # --- Pre-process image ----------------------------------------------
        try:
            img = Image.open(file.stream).convert("RGB")
        except Exception:
            return jsonify({"error": "Could not open file as an image."}), 400

        img = img.resize((256, 256))
        img_array = np.array(img, dtype=np.float32) / 255.0  # normalize 0-1
        img_array = np.expand_dims(img_array, axis=0)         # (1, 256, 256, 3)

        # --- Predict --------------------------------------------------------
        if model is not None:
            result = _run_inference(img_array)
        else:
            result = _stub_prediction()

        return jsonify(result), 200

    except Exception as e:
        print(f"❌ /predict/image error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print(f"🚀 Image service starting on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port)
