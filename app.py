import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "ppm"}
IMG_SIZE = 224
MODEL_PATH = "model/efficientnet_finetuned_final.keras"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Load model once at startup ──────────────────────────────────
model = load_model(MODEL_PATH)

# ── Class names ─────────────────────────────────────────────────
CLASS_NAMES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}


# ── Helper functions ────────────────────────────────────────────
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(img_path):
    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)  # shape (224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)
    # EfficientNet expects [0, 255] — no normalization
    return img_array


def predict(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)[0]  # shape (43,)
    top3_idx = np.argsort(predictions)[::-1][:3]  # top 3 indices
    results = []
    for idx in top3_idx:
        results.append(
            {
                "class_id": int(idx),
                "class_name": CLASS_NAMES[int(idx)],
                "confidence": float(predictions[idx]) * 100,
            }
        )
    return results


# ── Routes ──────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if "file" not in request.files:
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "" or not allowed_file(file.filename):
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    predictions = predict(filepath)

    return render_template("result.html", predictions=predictions, image_path=filepath)


if __name__ == "__main__":
    app.run(debug=True)
