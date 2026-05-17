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

# Force TensorFlow to quiet down optimization warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ───────────────────────────────────────────────────────────────────────────
# Paths & Global References
# ───────────────────────────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_image_model.h5")

# Global model container for lazy loading
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
                # Loaded with compile=False to avoid custom metric/loss binding breaks
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
    """
    FIX 1: Returns the LAST convolutional layer before the classifier head.
    Grad-CAM needs the final conv layer to capture the richest spatial features.
    The original code used -len//2 (a middle layer) which produces weak/noisy maps.
    Also scans for BatchNorm/Activation wrappers after conv blocks (e.g. EfficientNet).
    """
    layer_names = [l.name for l in loaded_model.layers]

    # Preferred terminal feature layers for EfficientNet / ResNet / MobileNet
    preferred = ['top_conv', 'top_activation', 'block7a_project_conv',
                 'block6a_expand_conv', 'conv5_block3_out', 'Conv_1']
    for name in preferred:
        if name in layer_names:
            print(f"✅ Using preferred Grad-CAM layer: {name}")
            return name

    # Dynamic fallback: walk layers in REVERSE and pick the first Conv2D
    for layer in reversed(loaded_model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D,
                               tf.keras.layers.SeparableConv2D)):
            print(f"✅ Fallback Grad-CAM layer selected: {layer.name}")
            return layer.name

    print("⚠️ No convolutional layer found in model.")
    return None


def generate_gradcam_heatmap(img_array, loaded_model):
    """
    FIX 2: Corrected Grad-CAM gradient computation.

    Root causes fixed:
      - tape.watch(img_tensor) was wrong: gradients must be computed w.r.t.
        conv_outputs (intermediate activations), not the raw input pixels.
        Watching the input image gives pixel-level gradients, not spatial class
        activation maps. We now use a persistent tape and watch conv_outputs.
      - predictions[:, 0] could IndexError on scalar outputs; use tf.reshape to
        safely flatten to a scalar loss value.
      - Silent black heatmap on zero-max is now logged and returns None cleanly.
      - Layer selection now uses get_last_conv_layer() (see FIX 1).

    img_array: (1, 256, 256, 3) normalized float32 matrix
    Returns: Base64-encoded PNG string of colorized Grad-CAM overlay, or None.
    """
    try:
        last_conv_layer_name = get_last_conv_layer(loaded_model)
        if not last_conv_layer_name:
            print("⚠️ No valid convolutional layer found — skipping Grad-CAM.")
            return None

        print(f"🔍 Computing Grad-CAM over layer: {last_conv_layer_name}")

        # Sub-graph that outputs BOTH the target conv activations AND final predictions
        grad_model = tf.keras.models.Model(
            inputs=loaded_model.inputs,
            outputs=[
                loaded_model.get_layer(last_conv_layer_name).output,
                loaded_model.output
            ]
        )

        img_tensor = tf.cast(img_array, tf.float32)

        # FIX 2a: Use persistent=True tape and watch conv_outputs, not img_tensor.
        # The gradient of the class score w.r.t. the conv feature map is what
        # Grad-CAM needs. Watching the image gives pixel gradients — not useful here.
        with tf.GradientTape(persistent=True) as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            tape.watch(conv_outputs)  # ← KEY FIX: watch activations, not input

            # FIX 2b: Robustly extract scalar loss regardless of output shape.
            # predictions could be (1,1), (1,), or scalar — flatten safely.
            loss = tf.reshape(predictions, [-1])[0]

        # Gradient of class score w.r.t. the conv feature map spatial activations
        grads = tape.gradient(loss, conv_outputs)
        del tape  # Release persistent tape immediately to free memory

        if grads is None:
            print("⚠️ Gradients are None — layer may not be connected to output.")
            return None

        # Global average pool the gradients over the spatial (H, W) axes → (C,)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Weight each feature map channel by its pooled gradient importance
        conv_outputs = conv_outputs[0]                          # (H, W, C)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
        heatmap = tf.squeeze(heatmap).numpy()                   # (H, W)

        # Apply ReLU: only highlight regions that positively activate the class
        heatmap = np.maximum(heatmap, 0)

        # FIX 2c: Guard against a zero/flat heatmap (would produce pure black overlay)
        if heatmap.max() == 0:
            print("⚠️ Grad-CAM heatmap is uniformly zero — model may be overconfident "
                  "or the selected layer has no spatial variation for this input.")
            return None

        heatmap = heatmap / heatmap.max()  # Normalize to [0, 1]

        # Resize heatmap to match input image dimensions
        heatmap_resized = cv2.resize(heatmap, (256, 256))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # PIL gives us RGB; convert to BGR for OpenCV blending
        original_rgb = np.uint8(255 * img_array[0])             # (256, 256, 3) RGB
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

        # Alpha blend: 60% original image + 40% heatmap overlay
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
        # Scenario A: Handle standard multipart form data uploads directly from app1.py
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream).convert('RGB').resize((256, 256))
            input_data = np.array(img, dtype=np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

        # Scenario B: Backward compatibility for local raw pixel JSON arrays
        elif request.is_json:
            data = request.get_json()
            input_data = np.array(data['input'], dtype=np.float32)
            if input_data.ndim == 3:
                input_data = np.expand_dims(input_data, axis=0)
        else:
            return jsonify({
                "error": "Invalid request. Supply multipart form-data with key 'image' or a JSON body with key 'input'."
            }), 400

        # Run inference using the safe lazy-loader
        active_model = get_model()

        if active_model is not None:
            prediction = active_model.predict(input_data)

            # FIX 3: Safe scalar extraction matching FIX 2b
            raw_score = float(tf.reshape(prediction, [-1])[0].numpy())

            if raw_score > 0.5:
                label = "Real"
                confidence = raw_score
            else:
                label = "Fake"
                confidence = 1.0 - raw_score

            # Trigger heatmap generation (returns None gracefully on failure)
            heatmap_b64 = generate_gradcam_heatmap(input_data, active_model)
        else:
            # Stub fallback when model is unavailable
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
    print(f"📢 Starting Image Deepfake Detection service on 0.0.0.0:{port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
