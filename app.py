# app.py
import os
import subprocess
from io import BytesIO

from flask import Flask, request, jsonify, send_file
import numpy as np
import cv2

from gpu_kernels import gpu_invert, gpu_brightness, gpu_contrast

app = Flask(__name__)


# -----------------------------------------------------------------------------
# Utility: decode uploaded image to BGR (OpenCV)
# -----------------------------------------------------------------------------

def decode_image(file_storage):
    """
    Takes a Werkzeug FileStorage (request.files['image']) and returns
    a BGR uint8 image (OpenCV format).
    """
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image. Unsupported format?")
    return img


def encode_image_bgr(img_bgr, ext=".jpg"):
    """
    Encode BGR image to bytes for HTTP response.
    """
    success, buf = cv2.imencode(ext, img_bgr)
    if not success:
        raise ValueError("Could not encode image.")
    return BytesIO(buf.tobytes())


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/api/transform/<op>", methods=["POST"])
def transform(op):
    """
    Apply a GPU-based transformation to an uploaded image.

    Example:
      curl -X POST -F "image=@input.jpg" http://localhost:5000/api/transform/invert --output out.jpg
      curl -X POST -F "image=@input.jpg" "http://localhost:5000/api/transform/brightness?shift=40" --output out.jpg
      curl -X POST -F "image=@input.jpg" "http://localhost:5000/api/transform/contrast?factor=1.2" --output out.jpg
    """
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file field"}), 400

    try:
        img = decode_image(request.files["image"])
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        if op == "invert":
            out = gpu_invert(img)

        elif op == "brightness":
            shift = int(request.args.get("shift", "40"))
            out = gpu_brightness(img, shift=shift)

        elif op == "contrast":
            factor = float(request.args.get("factor", "1.2"))
            out = gpu_contrast(img, factor=factor)

        else:
            return jsonify({"error": f"Unsupported operation '{op}'"}), 400

    except Exception as e:
        # In a real app, log the traceback
        return jsonify({"error": f"GPU processing failed: {e}"}), 500

    bio = encode_image_bgr(out, ext=".jpg")
    return send_file(bio, mimetype="image/jpeg")


@app.route("/api/info", methods=["GET"])
def info():
    """
    Returns basic GPU info + environment details.
    Nice for your résumé demo.
    """
    gpu_info = {}
    try:
        # Requires nvidia-smi to be available in the container/host
        result = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits"
        ]).decode("utf-8").strip()

        # If only one GPU, it's a single line
        name, mem_total, driver = [x.strip() for x in result.split(",")]
        gpu_info = {
            "name": name,
            "memory_total_mb": mem_total,
            "driver_version": driver,
        }
    except Exception as e:
        gpu_info = {"error": str(e)}

    return jsonify({
        "app": "GPU Image Transform Service",
        "gpu": gpu_info,
        "env": {
            "python_version": os.environ.get("PYTHON_VERSION", "unknown"),
            "cuda": "8.0 (expected on GTX 550 Ti demo)",
        }
    })


if __name__ == "__main__":
    # For local dev; in prod use gunicorn/uwsgi
    app.run(host="0.0.0.0", port=5000, debug=True)

