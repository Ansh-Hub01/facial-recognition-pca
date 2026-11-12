from pathlib import Path
from io import BytesIO
import threading

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, jsonify, render_template, request
from sklearn.datasets import fetch_olivetti_faces
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
MODEL_LOCK = threading.Lock()
MODEL = None
DATA = None

def load_and_train():
    """Train the same PCA + KNN approach used in the supplied project scripts."""
    global MODEL, DATA
    with MODEL_LOCK:
        if MODEL is not None:
            return MODEL, DATA

        faces = fetch_olivetti_faces(shuffle=True, random_state=42)
        X, y = faces.data, faces.target

        # Hold-out evaluation, matching the supplied project settings.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42, stratify=y
        )
        eval_pca = PCA(n_components=100, svd_solver="randomized", random_state=42)
        X_train_pca = eval_pca.fit_transform(X_train)
        X_test_pca = eval_pca.transform(X_test)
        eval_model = KNeighborsClassifier(n_neighbors=3)
        eval_model.fit(X_train_pca, y_train)
        predictions = eval_model.predict(X_test_pca)
        accuracy = float(accuracy_score(y_test, predictions))
        cm = confusion_matrix(y_test, predictions, labels=range(40))

        # Final recognition model is fitted on the complete dataset.
        pca = PCA(n_components=100, svd_solver="randomized", random_state=42)
        X_all_pca = pca.fit_transform(X)
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_all_pca, y)

        MODEL = (pca, model)
        DATA = {
            "faces": faces,
            "accuracy": accuracy,
            "confusion_matrix": cm.tolist(),
            "eigenvalues": pca.explained_variance_.tolist(),
            "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        }
        return MODEL, DATA

@app.route("/")
def home():
    return render_template("index.html")

@app.get("/api/status")
def status():
    try:
        _, data = load_and_train()
        return jsonify({
            "ready": True,
            "images": int(len(data["faces"].images)),
            "identities": int(len(np.unique(data["faces"].target))),
            "accuracy": data["accuracy"],
            "components": 100,
            "explained_variance": data["explained_variance"],
        })
    except Exception as exc:
        return jsonify({"ready": False, "error": str(exc)}), 503

@app.get("/api/metrics")
def metrics():
    try:
        _, data = load_and_train()
        cm = data["confusion_matrix"]
        diagonal = [cm[i][i] for i in range(40)]
        return jsonify({
            "accuracy": data["accuracy"],
            "confusion_matrix": cm,
            "eigenvalues": data["eigenvalues"],
            "correct_by_identity": diagonal,
            "identities": 40,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503

@app.post("/api/recognize")
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "Please choose an image to upload."}), 400
    file = request.files["image"]
    if not file.filename:
        return jsonify({"error": "Please choose an image file."}), 400
    try:
        image = Image.open(BytesIO(file.read()))
        image = ImageOps.exif_transpose(image).convert("L")
        # Olivetti images are 64x64 grayscale. Resize uploaded image to that format.
        image = ImageOps.fit(image, (64, 64), method=Image.Resampling.LANCZOS)
        vector = np.asarray(image, dtype=np.float32).reshape(1, -1) / 255.0
        (pca, model), data = load_and_train()
        projected = pca.transform(vector)
        distances, indices = model.kneighbors(projected, n_neighbors=3)
        prediction = int(model.predict(projected)[0])
        nearest = [
            {"identity": int(model._y[indices[0][i]]), "distance": float(distances[0][i])}
            for i in range(3)
        ]
        # Return normalized preview to the browser.
        import base64
        from PIL import Image as PILImage
        preview = BytesIO()
        image.save(preview, format="PNG")
        encoded = base64.b64encode(preview.getvalue()).decode("ascii")
        return jsonify({
            "success": True,
            "identity": prediction,
            "confidence_note": "KNN nearest-class result (not a probability or real-world identity).",
            "nearest_matches": nearest,
            "preview": "data:image/png;base64," + encoded,
            "model_accuracy": data["accuracy"],
        })
    except Exception as exc:
        return jsonify({"error": "Could not process this image. Please upload a valid JPG or PNG. Details: " + str(exc)}), 400

if __name__ == "__main__":
    # First launch downloads the Olivetti dataset if it is not already cached.
    app.run(host="127.0.0.1", port=5000, debug=True)
