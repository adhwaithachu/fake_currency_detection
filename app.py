"""
app.py — Flask Web App for Fake Currency Detection
Run: python app.py  →  open http://localhost:5000
"""

import os
import sys

from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename

# Ensure src/ is importable regardless of where app.py is run from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from predict import predict  # noqa: E402  (import after path fix)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED = {"png", "jpg", "jpeg", "bmp", "webp"}


def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not allowed(file.filename):
        return jsonify({"error": "Invalid file type. Use PNG, JPG, JPEG, BMP or WEBP."}), 400

    filename  = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(save_path)

    try:
        label, confidence, all_probs = predict(save_path)
        return jsonify({
            "label"     : label,
            "confidence": round(confidence * 100, 2),
            "all_probs" : {k: round(v * 100, 2) for k, v in all_probs.items()},
            "image_url" : url_for("static", filename=f"uploads/{filename}"),
        })
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("─" * 45)
    print(" Fake Currency Detection — Web App")
    print(" Open: http://localhost:5000")
    print("─" * 45)
    app.run(debug=True, host="0.0.0.0", port=5000)
