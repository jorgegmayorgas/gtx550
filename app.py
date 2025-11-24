# app.py
import io
from flask import Flask, request, send_file, jsonify
import numpy as np
import cv2

from gpu_kernels import gpu_invert, gpu_brightness, gpu_contrast

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "gpu-image-service",
        "status": "ok",
        "endpoints": [
            "/gpu/invert",
            "/gpu/brightness",
            "/gpu/contrast"
        ]
    })


def read_image_from_request():
    """Reads an uploaded image from Flask request."""
    if "image" not in request.files:
        return None, jsonify({"error": "No image file uploaded (field name: 'image')"}), 400

    file = request.files["image"]
    image_bytes = file.read()

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return None, jsonify({"error": "Invalid or unreadable image"}), 400

    return img, None, None


def send_image(img):
    """Encode and return the image as JPEG."""
    ok, buffer = cv2.imencode(".jpg", img)
    if not ok:
        return jsonify({"error": "Failed to encode output image"}), 500

    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype="image/jpeg",
        as_attachment=False
    )


@app.route("/gpu/invert", methods=["POST"])
def api_invert():
    img, error_response, status = read_image_from_request()
    if error_response:
        return error_response, status

    out = gpu_invert(img)
    return send_image(out)


@app.route("/gpu/brightness", methods=["POST"])
def api_brightness():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    data = file.read()

    # Leer imagen con OpenCV
    img_array = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    # Solo usamos beta como delta de brillo
    beta = int(request.form.get("beta", 0))  # por ejemplo -100..+100

    # Llamada correcta a tu kernel
    out = gpu_brightness(img, delta=beta)

    _, buf = cv2.imencode(".jpg", out)
    return send_file(
        io.BytesIO(buf.tobytes()),
        mimetype="image/jpeg",
    )

@app.route("/gpu/contrast", methods=["POST"])
def api_contrast():
    img, error_response, status = read_image_from_request()
    if error_response:
        return error_response, status

    alpha = float(request.form.get("alpha", 1.3))
    midpoint = float(request.form.get("midpoint", 127.0))

    out = gpu_contrast(img, alpha=alpha, midpoint=midpoint)
    return send_image(out)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        use_reloader=False,
        threaded=False
    )

