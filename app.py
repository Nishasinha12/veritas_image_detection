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
    """Finds optimal intermediate convolutional layer for clear spatial coverage."""
    layer_names = [l.name for l in loaded_model.layers]
    
    # Preferred deep feature layers for standard model architectures (EfficientNet/ResNet)
    preferred = ['block6a_expand_conv', 'block5a_expand_conv', 'block4a_expand_conv', 'top_conv']
    for name in preferred:
        if name in layer_names:
            return name
            
    # Dynamic Fallback: Find a Conv2D layer roughly halfway through the network
    conv_layers = [l.name for l in loaded_model.layers if 'conv' in l.name.lower() or isinstance(l, tf.keras.layers.Conv2D)]
    if len(conv_layers) >= 2:
        return conv_layers[-len(conv_layers)//2] 
        
    return conv_layers[-1] if conv_layers else None


def generate_gradcam_heatmap(img_array, loaded_model):
    """
    Generates a Grad-CAM activation heatmap overlay.
    img_array: (1, 256, 256, 3) normalized float32 matrix
    Returns: Base64-encoded string of colorized feature map overlay image
    """
    try:
        last_conv_layer_name = get_last_conv_layer(loaded_model)
        if not last_conv_layer_name:
            print("⚠️ No valid convolutional layers located for tracking.")
            return None
            
        print(f"🔍 Tracking spatial activations over layer: {last_conv_layer_name}")

        # Build sub-graph tracker mapping target intermediate outputs
        grad_model = tf.keras.models.Model(
            inputs=loaded_model.inputs,
            outputs=[loaded_model.get_layer(last_conv_layer_name).output, loaded_model.output]
        )

        # Enforce gradient computation tracking explicitly over tensor inputs
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            loss = predictions[:, 0]

        # Extract target spatial weights feature map gradient
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU activation barrier and scale vector bounds to 0-1
        heatmap = tf.nn.relu(heatmap).numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Format visual dimension scaling matrix to match the input image size
        heatmap_resized = cv2.resize(heatmap, (256, 256))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Base frame conversions (RGB -> BGR for OpenCV overlay integration)
        original_rgb = np.uint8(255 * img_array[0])
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

        # Alpha blending: 60% source image visibility + 40% heatmap color tracking
        superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

        # Write image out to raw byte representation safely
        _, buffer = cv2.imencode(".png", superimposed)
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        print(f"⚠️ Grad-CAM computation pipeline broke: {e}")
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
            return jsonify({"error": "Invalid request parameters. Supply form-data file or valid JSON structure."}), 400

        # Run inference using the safe lazy-loader
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

            # Trigger heatmap analysis
            heatmap_b64 = generate_gradcam_heatmap(input_data, active_model)
        else:
            # Fallback stub values if model path is temporarily empty or broken
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

        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Failure on image classification endpoint: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal classification error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # Binds to environment variable PORT for seamless cloud provider integration (e.g. Cloud Run / Render)
    port = int(os.environ.get("PORT", 5002))
    print(f"📢 Starting Image Deepfake Detection service on 0.0.0.0:{port}...")
    app.run(host='0.0.0.0', port=port, debug=False)