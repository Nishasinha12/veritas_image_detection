"""
Veritas — Image Deepfake Detection Microservice
================================================
Production-ready Flask service for image deepfake detection with Grad-CAM heatmaps.
Handles file uploads from app1.py and features lazy-loading to prevent cloud crashes.
"""

import os
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
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("⬇️  Downloading deepfake_image_model.h5 from Hugging Face...")
        try:
            hf_hub_download(
                repo_id="nishuu12/veritas-models",
                filename="deepfake_image_model.h5",
                local_dir=MODEL_DIR,
                token=os.getenv("HF_TOKEN")
            )
            print("✅ Image model downloaded successfully.")
        except Exception as e:
            print(f"⚠️  Hugging Face download failed: {e}")
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
                # Print layer summary to help debug layer name selection
                print("📋 Model layers:")
                for i, layer in enumerate(model.layers):
                    print(f"   [{i:3d}] {layer.__class__.__name__:30s} | {layer.name}")
            except Exception as e:
                print(f"⚠️  Failed to load model: {e}")
                traceback.print_exc()
                model = None
        else:
            print("ℹ️  Model file missing — running in stub fallback mode.")
    return model


# ───────────────────────────────────────────────────────────────────────────
# Layer Selection for Grad-CAM
# ───────────────────────────────────────────────────────────────────────────
def get_last_conv_layer(loaded_model):
    """
    Returns the name of the last spatial convolutional layer before the
    classifier head. This is the correct layer for Grad-CAM — it holds the
    richest class-discriminative spatial information.

    Strategy:
      1. Check a hardcoded list of known terminal layers for common architectures.
      2. Walk the layer list in REVERSE and return the first Conv2D / 
         DepthwiseConv2D / SeparableConv2D found.
      3. If nothing found, return None (heatmap will be skipped).
    """
    # Known terminal conv layer names across common backbone families
    preferred = [
        # EfficientNetB0-B7
        'top_conv', 'top_activation',
        'block7a_project_conv', 'block7a_expand_conv',
        'block6a_expand_conv', 'block6a_project_conv',
        # ResNet50 / ResNet101
        'conv5_block3_out', 'conv5_block3_3_conv',
        # MobileNetV2
        'Conv_1', 'block_16_project',
        # VGG / basic CNN
        'block5_conv3', 'block5_conv2',
        # Xception
        'block14_sepconv2_act',
        # InceptionV3
        'mixed10',
    ]

    layer_names = {l.name for l in loaded_model.layers}
    for name in preferred:
        if name in layer_names:
            print(f"✅ Grad-CAM: using preferred layer '{name}'")
            return name

    # Reverse-scan fallback: find the last convolutional layer
    conv_types = (
        tf.keras.layers.Conv2D,
        tf.keras.layers.DepthwiseConv2D,
        tf.keras.layers.SeparableConv2D,
    )
    for layer in reversed(loaded_model.layers):
        if isinstance(layer, conv_types):
            print(f"✅ Grad-CAM: fallback layer selected '{layer.name}'")
            return layer.name

    print("❌ Grad-CAM: no suitable convolutional layer found in model.")
    return None


# ───────────────────────────────────────────────────────────────────────────
# Grad-CAM Heatmap Generation
# ───────────────────────────────────────────────────────────────────────────
def generate_gradcam_heatmap(img_array, loaded_model):
    """
    Generates a Grad-CAM activation heatmap and blends it over the input image.

    THE KEY BUG THAT WAS BREAKING HEATMAPS (now fixed):
    ─────────────────────────────────────────────────────
    Previous versions called tape.watch(conv_outputs) AFTER grad_model() had
    already executed inside the same `with` block. TensorFlow's GradientTape
    only records operations on tensors it was watching AT THE TIME those
    operations ran. Watching after the forward pass = no gradient graph = None.

    Correct pattern (used here):
      1. Open a PERSISTENT tape.
      2. Cast the input to a tf.Variable or use tape.watch() BEFORE the call.
      3. Run the forward pass — tape records all ops on watched tensors.
      4. Define the scalar loss from predictions.
      5. Call tape.gradient(loss, conv_outputs) — this now works.
      6. Delete the persistent tape to free memory.

    img_array : np.ndarray  shape (1, 256, 256, 3), dtype float32, range [0,1]
    Returns   : str | None  — base64-encoded PNG of the heatmap overlay
    """
    try:
        last_conv_layer_name = get_last_conv_layer(loaded_model)
        if not last_conv_layer_name:
            return None

        print(f"🔍 Computing Grad-CAM over layer: '{last_conv_layer_name}'")

        # Build a sub-model that outputs both the chosen conv activations
        # and the final classifier output in a single forward pass.
        grad_model = tf.keras.models.Model(
            inputs=loaded_model.inputs,
            outputs=[
                loaded_model.get_layer(last_conv_layer_name).output,  # (1, H, W, C)
                loaded_model.output                                     # (1, 1) or (1,)
            ]
        )

        # ── THE FIX: persistent tape + watch BEFORE the forward pass ──────────
        #
        # Why persistent=True?
        #   We call tape.gradient() once for conv_outputs. Without persistent,
        #   the tape is consumed on the first gradient call and raises an error.
        #   With persistent, we can safely call it once and then delete the tape.
        #
        # Why watch BEFORE grad_model()?
        #   GradientTape only builds the computation graph for tensors it was
        #   watching when the ops that produced those tensors were executed.
        #   If you run grad_model() first, the activations are already computed
        #   outside the tape's awareness — gradients will be None.
        #
        #   We solve this by:
        #     a) Wrapping img_array in a tf.Variable (automatically watched), OR
        #     b) Watching the input tensor first, running the model, then watching
        #        conv_outputs for a second-order approach.
        #
        #   The cleanest pattern: watch the INPUT, run the model inside the tape,
        #   then compute d(loss)/d(conv_outputs) using the recorded graph.
        # ──────────────────────────────────────────────────────────────────────

        img_tensor = tf.Variable(img_array, trainable=False, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            # tape automatically watches tf.Variables; no explicit tape.watch needed
            conv_outputs, predictions = grad_model(img_tensor, training=False)

            # Safely extract scalar class score regardless of output shape
            # Handles: (1,1), (1,), scalar
            loss = tf.reshape(predictions, [-1])[0]

        # ── Gradient computation ───────────────────────────────────────────────
        # d(class_score) / d(conv_feature_map)
        grads = tape.gradient(loss, conv_outputs)   # shape: (1, H, W, C)
        del tape  # IMPORTANT: release persistent tape immediately

        if grads is None:
            print("⚠️  Gradients are None — check that the conv layer is in the "
                  "computation graph and connected to the output.")
            return None

        # Global-average-pool gradients over spatial dims → importance per channel
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # shape: (C,)

        # Weight each activation map by the channel's gradient importance
        conv_outputs_np = conv_outputs[0].numpy()               # (H, W, C)
        pooled_grads_np = pooled_grads.numpy()                  # (C,)

        # Equivalent to: sum over channels of (activation_map * weight)
        heatmap = np.einsum('hwc,c->hw', conv_outputs_np, pooled_grads_np)  # (H, W)

        # ReLU: discard negative contributions (regions suppressing the class)
        heatmap = np.maximum(heatmap, 0)

        # Guard: if all activations are zero the overlay would be pure black
        if heatmap.max() == 0:
            print("⚠️  Heatmap is uniformly zero — the selected layer may not have "
                  "spatial variance for this input (possible overconfident prediction).")
            return None

        # Normalize to [0, 1]
        heatmap /= heatmap.max()

        # ── Build overlay image ────────────────────────────────────────────────
        # Resize heatmap to match input resolution
        heatmap_resized = cv2.resize(heatmap, (256, 256))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Input img_array is RGB (from PIL); OpenCV works in BGR
        original_rgb = np.uint8(255 * img_array[0])
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

        # Alpha-blend: 60% original + 40% heatmap
        superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

        _, buffer = cv2.imencode(".png", superimposed)
        return base64.b64encode(buffer).decode("utf-8")

    except Exception as e:
        print(f"⚠️  Grad-CAM failed: {e}")
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
        # Scenario A: multipart file upload
        if 'image' in request.files:
            file = request.files['image']
            img  = Image.open(file.stream).convert('RGB').resize((256, 256))
            input_data = np.expand_dims(
                np.array(img, dtype=np.float32) / 255.0, axis=0
            )

        # Scenario B: raw JSON pixel array (backward-compatible)
        elif request.is_json:
            data       = request.get_json()
            input_data = np.array(data['input'], dtype=np.float32)
            if input_data.ndim == 3:
                input_data = np.expand_dims(input_data, axis=0)
        else:
            return jsonify({
                "error": "Invalid request. Send multipart/form-data with key 'image', "
                         "or application/json with key 'input'."
            }), 400

        active_model = get_model()

        if active_model is not None:
            raw_preds  = active_model.predict(input_data)
            raw_score  = float(tf.reshape(raw_preds, [-1])[0].numpy())

            if raw_score > 0.5:
                label      = "Real"
                confidence = raw_score
            else:
                label      = "Fake"
                confidence = 1.0 - raw_score

            heatmap_b64 = generate_gradcam_heatmap(input_data, active_model)
        else:
            print("⚠️  Model not initialized — returning stub response.")
            label       = "Fake"
            confidence  = 0.85
            heatmap_b64 = None

        response_data = {
            "prediction": label,
            "confidence": round(confidence, 4),
        }

        if heatmap_b64:
            response_data["heatmap_base64"] = heatmap_b64
        else:
            response_data["heatmap_note"] = "Heatmap unavailable for this input."

        return jsonify(response_data), 200

    except Exception as e:
        print(f"❌ Endpoint error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Internal classification error: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    print(f"📢 Starting Veritas Image Service on 0.0.0.0:{port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
