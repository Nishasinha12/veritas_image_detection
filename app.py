from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import sys
import base64
import cv2
import traceback
from PIL import Image

app = Flask(__name__)

# --- MODEL LOADING ---
MODEL_PATH = os.path.join("models", "deepfake_image_model.h5")
print(f"🚀 Attempting to load image model from: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Image deepfake model loaded successfully.")
    print("📋 Model layers (last 10):")
    for layer in model.layers[-10:]:
        print(f"   {layer.name} -> {type(layer).__name__}")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load image model.")
    print(f"Error details: {e}")
    sys.exit(1)


def get_last_conv_layer(loaded_model):
    """Finds optimal conv layer for Grad-CAM."""
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
    Grad-CAM heatmap generation.
    img_array: (1, 256, 256, 3) normalized float32
    Returns: base64 PNG string or None
    """
    try:
        last_conv_layer_name = get_last_conv_layer(loaded_model)
        if not last_conv_layer_name:
            print("⚠️ No valid conv layer found.")
            return None

        print(f"🔍 Computing Grad-CAM over layer: {last_conv_layer_name}")

        img_tensor = tf.cast(img_array, tf.float32)

        # Single grad_model: input → [conv_output, final_output]
        grad_model = tf.keras.models.Model(
            inputs=loaded_model.inputs,
            outputs=[
                loaded_model.get_layer(last_conv_layer_name).output,
                loaded_model.output
            ]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            tape.watch(conv_outputs)   # ✅ watch AFTER getting conv_outputs,
            loss = tf.identity(predictions[:, 0])  # ✅ tf.identity keeps graph connection

        grads = tape.gradient(loss, conv_outputs)
        print(f"🔬 Grads: {grads}")

        if grads is None:
            # Fallback: saliency map using input gradients
            print("⚠️ Grad-CAM grads None — using input saliency fallback.")
            with tf.GradientTape() as tape2:
                tape2.watch(img_tensor)
                _, predictions2 = grad_model(img_tensor, training=False)
                loss2 = predictions2[:, 0]
            grads2 = tape2.gradient(loss2, img_tensor)
            if grads2 is None:
                print("❌ Fallback gradients also None.")
                return None
            heatmap = tf.reduce_mean(tf.abs(grads2), axis=-1)[0].numpy()
        else:
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_out = conv_outputs[0]
            heatmap = (conv_out @ pooled_grads[..., tf.newaxis])
            heatmap = tf.squeeze(heatmap).numpy()

        heatmap = np.maximum(heatmap, 0)

        if heatmap.max() == 0:
            print("⚠️ Heatmap is uniformly zero.")
            return None

        heatmap = heatmap / heatmap.max()

        heatmap_resized = cv2.resize(heatmap, (256, 256))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        original_rgb = np.uint8(255 * img_array[0])
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

        _, buffer = cv2.imencode(".png", superimposed)
        result = base64.b64encode(buffer).decode("utf-8")
        print(f"✅ Heatmap generated, size: {len(result)} chars")
        return result

    except Exception as e:
        print(f"⚠️ Grad-CAM failed: {e}")
        traceback.print_exc()
        return None


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
        # ✅ Scenario A: multipart form-data (from app1.py)
        if 'image' in request.files:
            file = request.files['image']
            img = Image.open(file.stream).convert('RGB').resize((256, 256))
            input_data = np.array(img, dtype=np.float32) / 255.0
            input_data = np.expand_dims(input_data, axis=0)

        # ✅ Scenario B: raw JSON array (local testing)
        elif request.is_json:
            data = request.get_json()
            input_data = np.array(data['input'], dtype=np.float32)
            if input_data.ndim == 3:
                input_data = np.expand_dims(input_data, axis=0)

        else:
            return jsonify({"error": "Supply multipart 'image' or JSON 'input'."}), 400

        prediction = model.predict(input_data)
        raw_score = float(prediction[0][0])

        if raw_score > 0.5:
            label = "Real"
            confidence = raw_score
        else:
            label = "Fake"
            confidence = 1.0 - raw_score

        heatmap_b64 = generate_gradcam_heatmap(input_data, model)
        print(f"🗺️ Heatmap generated: {heatmap_b64 is not None}")

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
        print(f"❌ Image prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5002))
    print(f"📢 Starting Image service on 0.0.0.0:{port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
