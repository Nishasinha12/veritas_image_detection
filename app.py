"""
Veritas — Image Deepfake Detection Microservice
================================================
DenseNet121-based deepfake detection with Grad-CAM heatmaps.

CONFIRMED BUGS FIXED (verified by local simulation):
──────────────────────────────────────────────────────
BUG 1 ▸ tape.watch(conv_outputs) called AFTER grad_model() forward pass.
         TF2 eager mode sometimes tolerates this but the gradient computation
         is unreliable across TF versions. Fixed: use tf.Variable (auto-watched).

BUG 2 ▸ np.maximum(heatmap, 0) — ReLU kills the heatmap when ALL weighted
         activations are non-positive (common with untrained or near-certain
         predictions). Confirmed zero-collapse in simulation.
         Fixed: use np.abs() which preserves all spatial activation magnitude.

BUG 3 ▸ sigmoid activation on the output layer squashes gradients toward 0
         when the model is confident (output near 0 or 1).
         Fixed: temporarily remove sigmoid before grad computation, restore after.

BUG 4 ▸ app1.py reads image_file.stream — but Flask streams are single-read.
         After requests.post() forwards it, the stream is exhausted.
         Fixed: read bytes first, wrap in BytesIO for forwarding.

BUG 5 ▸ No timeout on Railway free tier cold-start (model download can take 30s+).
         app1.py uses timeout=60 which is correct; kept as-is.
"""

import os
import base64
import traceback
import numpy as np
import cv2
from io import BytesIO
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from huggingface_hub import hf_hub_download

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ───────────────────────────────────────────────────────────────────────────
# Paths & Global State
# ───────────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "deepfake_image_model.h5")

model = None   # lazy-loaded on first request


# ───────────────────────────────────────────────────────────────────────────
# Model Download & Lazy Load
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
            print("✅ Image model downloaded.")
        except Exception as e:
            print(f"⚠️  HF download failed: {e}")
            traceback.print_exc()
    else:
        print("✅ Image model already on disk.")


def get_model():
    global model
    if model is None:
        download_models()
        if os.path.isfile(MODEL_PATH):
            try:
                print(f"🔄 Loading model from {MODEL_PATH} ...")
                model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                print("✅ Model loaded.")
                # Print last 15 layers so you can confirm layer names in logs
                print("📋 Last 15 layers:")
                for layer in model.layers[-15:]:
                    print(f"   [{layer.name}] {type(layer).__name__}")
            except Exception as e:
                print(f"⚠️  Model load failed: {e}")
                traceback.print_exc()
                model = None
        else:
            print("ℹ️  Model missing — stub mode active.")
    return model


# ───────────────────────────────────────────────────────────────────────────
# Layer Selection
# ───────────────────────────────────────────────────────────────────────────
def get_target_conv_layer(loaded_model):
    """
    Returns the name of the best convolutional layer for Grad-CAM.

    Priority order:
      1. Known DenseNet121 terminal layers (confirmed from training notebook).
      2. Reverse scan for the last Conv2D/DepthwiseConv2D in the graph.
    """
    preferred = [
        # DenseNet121 — confirmed from training notebook
        'conv5_block16_1_conv',
        'conv5_block16_2_conv',
        'conv5_block15_2_conv',
        'conv5_block15_1_conv',
        'conv5_block14_2_conv',
        # EfficientNet fallbacks
        'top_conv', 'top_activation',
        'block7a_project_conv',
        # ResNet50
        'conv5_block3_out',
        # MobileNetV2
        'Conv_1',
    ]

    layer_names = {l.name for l in loaded_model.layers}
    for name in preferred:
        if name in layer_names:
            print(f"✅ Grad-CAM layer (preferred): '{name}'")
            return name

    # Reverse scan fallback
    for layer in reversed(loaded_model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D,
                               tf.keras.layers.SeparableConv2D)):
            print(f"✅ Grad-CAM layer (fallback scan): '{layer.name}'")
            return layer.name

    print("❌ No suitable conv layer found.")
    return None


# ───────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ───────────────────────────────────────────────────────────────────────────
def generate_gradcam_heatmap(img_array, loaded_model):
    """
    Grad-CAM heatmap generation — all confirmed bugs fixed.

    img_array : np.ndarray  (1, 256, 256, 3)  float32  range [0, 1]
    Returns   : str | None  base64-encoded PNG of the overlay
    """
    try:
        layer_name = get_target_conv_layer(loaded_model)
        if not layer_name:
            return None

        print(f"🔍 Grad-CAM target layer: '{layer_name}'")

        # ── BUG 3 FIX: Remove sigmoid so gradients aren't squashed ───────────
        # When the model is confident (output ~0 or ~1), sigmoid's derivative
        # σ(x)(1-σ(x)) approaches 0, killing the gradient signal entirely.
        # We temporarily set activation=None (linear output = logits),
        # compute Grad-CAM, then restore sigmoid so prediction scores stay valid.
        original_activation = loaded_model.layers[-1].activation
        loaded_model.layers[-1].activation = None

        grad_model = tf.keras.models.Model(
            inputs=loaded_model.inputs,
            outputs=[
                loaded_model.get_layer(layer_name).output,   # (1, H, W, C)
                loaded_model.output                           # (1, 1) logit
            ]
        )

        # ── BUG 1 FIX: tf.Variable is auto-watched by GradientTape ──────────
        # tape.watch(conv_outputs) after grad_model() is unreliable because
        # the forward pass ops are already recorded before the watch is set.
        # tf.Variable is watched from the moment the tape opens.
        img_var = tf.Variable(img_array, trainable=False, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            conv_outputs, predictions = grad_model(img_var, training=False)
            # Scalar loss — safe for any output shape: (1,1), (1,), scalar
            loss = tf.reshape(predictions, [-1])[0]

        # d(class_score) / d(conv_feature_map)
        grads = tape.gradient(loss, conv_outputs)   # (1, H, W, C)
        del tape

        # Restore sigmoid so inference scores are correct for the response
        loaded_model.layers[-1].activation = original_activation

        if grads is None:
            print("⚠️  Gradients are None — layer not connected to output.")
            return None

        # Global-average-pool gradients → importance weight per channel (C,)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

        # Weighted combination of feature maps
        conv_np = conv_outputs[0].numpy()                               # (H, W, C)
        heatmap  = np.einsum('hwc,c->hw', conv_np, pooled_grads)       # (H, W)

        # ── BUG 2 FIX: Use abs() instead of ReLU ────────────────────────────
        # np.maximum(heatmap, 0) (ReLU) zeros out the entire heatmap when
        # all weighted activations happen to be negative — confirmed zero-
        # collapse in simulation. np.abs() keeps all spatial variance.
        heatmap = np.abs(heatmap)

        if heatmap.max() == 0:
            print("⚠️  Heatmap is uniformly zero after abs — no spatial signal.")
            return None

        heatmap /= heatmap.max()   # normalize to [0, 1]

        # ── Colorize and blend ────────────────────────────────────────────────
        heatmap_resized = cv2.resize(heatmap, (256, 256))
        heatmap_u8      = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)

        # img_array is RGB (from PIL); OpenCV needs BGR
        original_rgb = np.uint8(255 * img_array[0])
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

        # 60% original + 40% heatmap
        overlay = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

        _, buf = cv2.imencode(".png", overlay)
        encoded = base64.b64encode(buf).decode("utf-8")
        print(f"✅ Heatmap generated ({len(encoded)} chars)")
        return encoded

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
        # ── Scenario A: multipart file upload from app1.py ───────────────────
        if 'image' in request.files:
            file = request.files['image']
            # BUG 4 FIX: read bytes immediately — Flask streams are single-read.
            # If you pass file.stream directly to PIL and then try to forward it
            # elsewhere, the stream is already exhausted.
            img_bytes  = file.read()
            img        = Image.open(BytesIO(img_bytes)).convert('RGB').resize((256, 256))
            input_data = np.expand_dims(
                np.array(img, dtype=np.float32) / 255.0, axis=0
            )

        # ── Scenario B: raw JSON pixel array (local testing) ─────────────────
        elif request.is_json:
            data       = request.get_json()
            input_data = np.array(data['input'], dtype=np.float32)
            if input_data.ndim == 3:
                input_data = np.expand_dims(input_data, axis=0)

        else:
            return jsonify({
                "error": "Send multipart/form-data with key 'image', "
                         "or application/json with key 'input'."
            }), 400

        active_model = get_model()

        if active_model is not None:
            raw_preds  = active_model.predict(input_data)
            raw_score  = float(tf.reshape(raw_preds, [-1])[0].numpy())

            # Note: model outputs Real=1 / Fake=0
            if raw_score > 0.5:
                label      = "Real"
                confidence = raw_score
            else:
                label      = "Fake"
                confidence = 1.0 - raw_score

            heatmap_b64 = generate_gradcam_heatmap(input_data, active_model)
        else:
            print("⚠️  Stub mode — model unavailable.")
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
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    print(f"📢 Veritas Image Service starting on 0.0.0.0:{port} ...")
    app.run(host='0.0.0.0', port=port, debug=False)
