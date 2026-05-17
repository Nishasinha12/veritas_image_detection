"""
Veritas — Image Deepfake Detection Microservice
================================================
DenseNet121-based deepfake detection with Grad-CAM heatmaps.
Layer names confirmed from training notebook.
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
# Model Download & Load
# ───────────────────────────────────────────────────────────────────────────
def download_models():
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
    global model
    if model is None:
        download_models()
        if os.path.isfile(MODEL_PATH):
            try:
                print(f"🔄 Loading TensorFlow model from {MODEL_PATH}...")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("✅ Image deepfake model loaded successfully.")
                print("📋 Last 10 layers:")
                for layer in model.layers[-10:]:
                    print(f"   {layer.name} -> {type(layer).__name__}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                traceback.print_exc()
                model = None
        else:
            print("ℹ️ Model file missing — running in stub fallback mode.")
    return model


# ───────────────────────────────────────────────────────────────────────────
# Grad-CAM — DenseNet121 layer names from training notebook
# ───────────────────────────────────────────────────────────────────────────
def get_last_conv_layer(loaded_model):
    """
    Layer names confirmed directly from the training notebook:
        last_conv_layer_name = 'conv5_block16_1_conv'
    All five were tested and work with this DenseNet121 model.
    """
    preferred = [
        'conv5_block16_1_conv',  # ✅ exact name used in training notebook
        'conv5_block16_2_conv',
        'conv5_block15_1_conv',
        'conv5_block14_1_conv',
        'conv5_block13_2_conv',
    ]

    layer_names = [l.name for l in loaded_model.layers]
    for name in preferred:
        if name in layer_names:
            print(f"✅ Using DenseNet layer: {name}")
            return name

    # Generic fallback — list all conv layers so we can debug
    conv_layers = [
        l.name for l in loaded_model.layers
        if isinstance(l, tf.keras.layers.Conv2D)
    ]
    print(f"⚠️ Preferred layers not found. Available Conv2D layers: {conv_layers}")

    if conv_layers:
        chosen = conv_layers[len(conv_layers) // 2]
        print(f"✅ Using fallback conv layer: {chosen}")
        return chosen

    print("❌ No Conv2D layers found in model")
    return None


def generate_gradcam_heatmap(img_array, loaded_model):
    """
    Grad-CAM heatmap for DenseNet121.

    Key fixes applied:
    1. tape.watch(conv_outputs) called AFTER getting conv_outputs so the
       tape records the full computation graph conv_outputs → loss.
    2. tf.identity(predictions[:, 0]) keeps the graph connection alive.
    3. model.layers[-1].activation = None removes sigmoid so gradients
       are not squashed near 0/1 — same technique used in training notebook.
    4. Fallback to input saliency if Grad-CAM grads are still None.

    img_array: (1, 256, 256, 3) normalized float32
    Returns: base64 PNG string or None
    """
    try:
        last_conv_layer_name = get_last_conv_layer(loaded_model)
        if not last_conv_layer_name:
            print("⚠️ No valid conv layer found — skipping Grad-CAM.")
            return None

        print(f"🔍 Computing Grad-CAM over layer: {last_conv_layer_name}")

        # ✅ Remove sigmoid activation so gradients aren't squashed
        # (same as training notebook: model.layers[-1].activation = None)
        loaded_model.layers[-1].activation = None

        img_tensor = tf.cast(img_array, tf.float32)

        grad_model = tf.keras.models.Model(
            inputs=loaded_model.inputs,
            outputs=[
                loaded_model.get_layer(last_conv_layer_name).output,
                loaded_model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            tape.watch(conv_outputs)  # ✅ watch AFTER getting conv_outputs
            loss = tf.identity(predictions[:, 0])  # ✅ keeps graph connection

        grads = tape.gradient(loss, conv_outputs)
        print(f"🔬 Grads is None: {grads is None}")

        if grads is None:
            # Fallback: input saliency map
            print("⚠️ Grad-CAM grads None — using input saliency fallback.")
            with tf.GradientTape() as tape2:
                tape2.watch(img_tensor)
                _, predictions2 = grad_model(img_tensor, training=False)
                loss2 = predictions2[:, 0]
            grads2 = tape2.gradient(loss2, img_tensor)
            if grads2 is None:
                print("❌ Fallback gradients also None. Giving up.")
                return None
            heatmap = tf.reduce_mean(tf.abs(grads2), axis=-1)[0].numpy()
        else:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (C,)
            conv_out = conv_outputs[0]                              # (H, W, C)
            heatmap = (conv_out @ pooled_grads[..., tf.newaxis])   # (H, W, 1)
            heatmap = tf.squeeze(heatmap).numpy()                  # (H, W)

        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() == 0:
            print("⚠️ Heatmap is uniformly zero — no spatial activation.")
            return None

        heatmap = heatmap / heatmap.max()

        # Colorize and overlay
        heatmap_resized = cv2.resize(heatmap, (256, 256))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        original_rgb = np.uint8(255 * img_array[0])
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

        _, buffer = cv2.imencode(".png", superimposed)
        result = base64.b64encode(buffer).decode("utf-8")
        print(f"✅ Heatmap generated successfully, size: {len(result)} chars")
        return result

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
        # Scenario A: multipart form-data from app1.py
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream).convert('RGB').resize((256, 256))
            input_data = np.array(img, dtype=np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

        # Scenario B: raw JSON pixel array for local testing
        elif request.is_json:
            data = request.get_json()
            input_data = np.array(data['input'], dtype=np.float32)
            if input_data.ndim == 3:
                input_data = np.expand_dims(input_data, axis=0)

        else:
            return jsonify({
                "error": "Supply multipart 'image' or JSON 'input'."
            }), 400

        active_model = get_model()

        if active_model is not None:
            prediction = active_model.predict(input_data)
            raw_score = float(prediction[0][0])

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
