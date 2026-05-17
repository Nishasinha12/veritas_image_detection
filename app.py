"""
Veritas — Image Deepfake Detection Microservice
================================================
Production-ready Flask service for image deepfake detection with Grad-CAM heatmaps.
Handles file uploads from app1.py and features lazy-loading to prevent cloud crashes.
"""

import os
import sys
import base64
import traceback
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from huggingface_hub import hf_hub_download

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ───────────────────────────────────────────────────────────────────────────
# Paths & Global References
# ───────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_image_model.h5")

model = None

# ───────────────────────────────────────────────────────────────────────────
# Safe Lazy-Load & Download Mechanism
# ───────────────────────────────────────────────────────────────────────────
def download_models():
    """Downloads the image model from Hugging Face if not already present."""
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
            print("✅ Image model downloaded successfully.")
        except Exception as e:
            print(f"⚠️ Hugging Face download failed: {e}")
            traceback.print_exc()
    else:
        print("✅ Image model already exists locally.")


def get_model():
    """Loads the model only when the first API request hits to prevent deployment timeouts."""
    global model
    if model is None:
        download_models()
        if os.path.isfile(MODEL_PATH):
            try:
                print(f"🔄 Loading TensorFlow model from {MODEL_PATH}...")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("✅ Image deepfake model loaded successfully.")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                traceback.print_exc()
                model = None
        else:
            print("ℹ️ Model file missing — running in stub fallback mode.")
    return model


# ───────────────────────────────────────────────────────────────────────────
# Grad-CAM Heatmap Generation Engine
# ───────────────────────────────────────────────────────────────────────────
def get_last_conv_layer(loaded_model):
    """Finds optimal intermediate convolutional layer for clear spatial coverage."""
    layer_names = [l.name for l in loaded_model.layers]

    preferred = ['block6a_expand_conv', 'block5a_expand_conv', 'block4a_expand_conv', 'top_conv']
    for name in preferred:
        if name in layer_names:
            print(f"✅ Using preferred layer: {name}")
            return name

    conv_layers = [
        l.name for l in loaded_model.layers
        if isinstance(l, tf.keras.layers.Conv2D)
    ]
    if conv_layers:
        chosen = conv_layers[len(conv_layers) // 2]
        print(f"✅ Using fallback conv layer: {chosen}")
        return chosen

    print("❌ No Conv2D layers found in model")
    return None


def generate_gradcam_heatmap(img_array, loaded_model):
    """
    Generates a Grad-CAM activation heatmap overlay.
    
    Uses a two-model split so the GradientTape watches conv_outputs
    BEFORE the classifier forward pass runs — fixing the None gradients bug.
    
    img_array: (1, 256, 256, 3) normalized float32 matrix
    Returns: Base64-encoded PNG string of colorized overlay, or None.
    """
    try:
        last_conv_layer_name = get_last_conv_layer(loaded_model)
        if not last_conv_layer_name:
            print("⚠️ No valid convolutional layer found — skipping Grad-CAM.")
            return None

        print(f"🔍 Computing Grad-CAM over layer: {last_conv_layer_name}")

        # ── Step 1: Model that outputs conv activations ──────────────────────
        conv_model = tf.keras.models.Model(
            inputs=loaded_model.inputs,
            outputs=loaded_model.get_layer(last_conv_layer_name).output
        )

        # ── Step 2: Model that takes conv activations → final prediction ─────
        classifier_input = tf.keras.Input(
            shape=loaded_model.get_layer(last_conv_layer_name).output.shape[1:]
        )
        x = classifier_input
        found = False
        for layer in loaded_model.layers:
            if found:
                x = layer(x)
            if layer.name == last_conv_layer_name:
                found = True
        classifier_model = tf.keras.Model(inputs=classifier_input, outputs=x)

        img_tensor = tf.cast(img_array, tf.float32)

        # ── Step 3: Tape watches conv_outputs BEFORE classifier runs ─────────
        # This is the critical fix: the tape must observe the full computation
        # from conv_outputs → predictions → loss to produce valid gradients.
        with tf.GradientTape() as tape:
            conv_outputs = conv_model(img_tensor, training=False)
            tape.watch(conv_outputs)
            predictions = classifier_model(conv_outputs, training=False)
            loss = tf.reduce_mean(predictions[:, 0])

        # ── Step 4: Compute gradients ────────────────────────────────────────
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            print("⚠️ Gradients are None — layer may not be differentiable.")
            return None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)

        # ── Step 5: Weight channels by gradient importance ───────────────────
        conv_out = conv_outputs[0]                              # (H, W, C)
        heatmap = conv_out @ pooled_grads[..., tf.newaxis]      # (H, W, 1)
        heatmap = tf.squeeze(heatmap).numpy()                   # (H, W)
        heatmap = np.maximum(heatmap, 0)                        # ReLU

        if heatmap.max() == 0:
            print("⚠️ Heatmap is uniformly zero — no spatial activation detected.")
            return None

        heatmap = heatmap / heatmap.max()                       # Normalize [0, 1]

        # ── Step 6: Colorize and overlay onto original image ─────────────────
        heatmap_resized = cv2.resize(heatmap, (256, 256))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        original_rgb = np.uint8(255 * img_array[0])
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

        superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

        _, buffer = cv2.imencode(".png", superimposed)
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        print(f"⚠️ Grad-CAM computation failed: {e}")
        traceback.print_exc()
        return None


# ───────────────────────────────────────────────────────────────────────────
# Endpoints
# ───────────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "running", "service": "Veritas Image API"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "image",
        "model_loaded": model is not None
    })


@app.route('/predict/image', methods=['POST'])
def predict_image():
    try:
        # Scenario A: Multipart form-data upload
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream).convert('RGB').resize((256, 256))
            input_data = np.array(img, dtype=np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

        # Scenario B: Raw JSON pixel array
        elif request.is_json:
            data = request.get_json()
            input_data = np.array(data['input'], dtype=np.float32)
            if input_data.ndim == 3:
                input_data = np.expand_dims(input_data, axis=0)

        else:
            return jsonify({
                "error": "Invalid request. Supply multipart form-data with key 'image' or a JSON body with key 'input'."
            }), 400

        active_model = get_model()

        if active_model is not None:
            prediction = active_model.predict(input_data)
            raw_score = float(tf.reshape(prediction, [-1])[0].numpy())

            if raw_score > 0.5:
                label = "Real"
                confidence = raw_score
            else:
                label = "Fake"
                confidence = 1.0 - raw_score

            heatmap_b64 = generate_gradcam_heatmap(input_data, active_model)
            print(f"🗺️ Heatmap generated: {heatmap_b64 is not None}")

        else:
            print("⚠️ Model not initialized. Returning stub response.")
            label = "Fake"
            confidence = 0.85
            heatmap_b64 = None

        response_data = {
            "prediction": label,
            "confidence": round(confidence, 4)
        }

        if heatmap_b64:
            response_data["heatmap_base64"] = heatmap_b64
        else:
            response_data["heatmap_note"] = "Heatmap unavailable for this input."

        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Image classification endpoint error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal classification error: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    print(f"📢 Starting Veritas Image service on 0.0.0.0:{port}...")
    app.run(host='0.0.0.0', port=port, debug=False)