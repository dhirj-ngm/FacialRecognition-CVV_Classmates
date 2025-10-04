# app.py
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import io
import base64
import os

from mediapipe_detection import detect_faces_mediapipe
from facial_recognition import get_insightface_embeddings_with_augmentation, match_embedding_to_database
from utils import load_pickle

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Limit uploads
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

DB_PATH = 'data/face_database.pkl'
RECOGNITION_THRESHOLD = 0.6  # adjust (0.5-0.7 typical) based on experiments

# try to load database at startup
face_db = {}
if os.path.exists(DB_PATH):
    try:
        face_db = load_pickle(DB_PATH)
        print("Loaded face database with labels:", list(face_db.keys())[:10])
    except Exception as e:
        print("Failed to load database:", e)
        face_db = {}
else:
    print("Face database not found at", DB_PATH, "- run build_database.py first.")


@app.route('/')
def index():
    return render_template('index.html')


def draw_boxes_and_labels(img, faces_info, recognitions):
    # faces_info: list of mediapipe faces with bbox [x,y,w,h]
    # recognitions: list of dicts: {'bbox': [x,y,w,h], 'name': name, 'score': float}
    out = img.copy()
    for rec in recognitions:
        x, y, w, h = rec['bbox']
        name = rec.get('name', 'unknown')
        score = rec.get('score', None)
        # box
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # label background
        label = f"{name}"
        if score is not None:
            label += f" ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x, y - th - 8), (x + tw + 6, y), (0, 255, 0), -1)
        cv2.putText(out, label, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out


@app.route('/predict', methods=['POST'])
def predict():
    global face_db
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "No file received"}), 400

    if not face_db:
        return jsonify({"error": "Face database not found. Run build_database.py to create data/face_database.pkl"}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Failed to decode image"}), 400

    # 1) mediapipe detection for boxes
    mp_faces = detect_faces_mediapipe(img)
    mp_boxes = [f['bbox'] for f in mp_faces]

    # 2) insightface: get embeddings for each detected face (support multiple faces)
    insight_faces = get_insightface_embeddings_with_augmentation(img, n_augment=2)  # fewer augmentations for speed
    # insight_faces is list of dicts {'bbox': [xmin,ymin,w,h], 'embeddings': np.array (k,D)}

    recognitions = []
    for f in insight_faces:
        bbox = f.get('bbox', [0,0,0,0])
        emb_arr = f.get('embeddings')  # shape (k,D)
        if emb_arr is None or emb_arr.size == 0:
            name, score = "unknown", 0.0
        else:
            # average embeddings across augmentations for this face
            emb_mean = np.mean(emb_arr, axis=0)
            # ensure normalized
            if np.linalg.norm(emb_mean) > 0:
                emb_mean = emb_mean / np.linalg.norm(emb_mean)
            name, score = match_embedding_to_database(emb_mean, face_db, threshold=RECOGNITION_THRESHOLD)
        recognitions.append({'bbox': bbox, 'name': name, 'score': score})

    # 3) annotated image: draw boxes and names (use mediapipe boxes if available else insight boxes)
    draw_boxes = [r['bbox'] for r in recognitions] if recognitions else mp_boxes
    annotated = draw_boxes_and_labels(img, mp_faces, recognitions)

    _, buffer = cv2.imencode('.png', annotated)
    png_b64 = base64.b64encode(buffer).decode('utf-8')

    # Prepare JSON output (compact)
    out_insight = []
    for r in recognitions:
        out_insight.append({
            'bbox': r['bbox'],
            'name': r['name'],
            'score': float(r['score'])
        })

    response = {
        'mediapipe_num_faces': len(mp_faces),
        'insightface_num_faces': len(insight_faces),
        'recognized': out_insight,
        'annotated_image_base64': png_b64
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)