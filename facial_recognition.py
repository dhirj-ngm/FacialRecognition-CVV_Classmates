# facial_recognition.py
import cv2
import numpy as np
import albumentations as A
from insightface.app import FaceAnalysis

# Try to initialize InsightFace (GPU if available, else CPU)
app = FaceAnalysis(name="buffalo_l")
try:
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("InsightFace prepared with GPU (ctx_id=0).")
except Exception as e:
    try:
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("InsightFace prepared with CPU (ctx_id=-1).")
    except Exception as e2:
        raise RuntimeError("Failed to prepare InsightFace (GPU or CPU).") from e2

# Augmentation pipeline (safe, small augmentations)
augmentor = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.4),
    A.Rotate(limit=10, p=0.4),
    A.Affine(translate_percent=0.05, scale=(0.95, 1.05), rotate=(-10, 10), p=0.4),
    A.GaussNoise(p=0.2),
])

def _normalize_embedding(emb):
    emb = np.array(emb, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb = emb / norm
    return emb

def augment_image(img, n=3):
    outs = []
    for _ in range(n):
        augmented = augmentor(image=img)["image"]
        outs.append(augmented)
    return outs

def get_insightface_faces(img):
    """
    Returns the insightface result list (faces). Each face object has .bbox and .normed_embedding
    """
    faces = app.get(img)
    return faces if faces else []

def get_single_face_embedding(img, use_augment=False, n_aug=3):
    """
    Return a normalized 1-D embedding for the largest face (or first face).
    If use_augment=True, also compute embeddings on augmentations and average them.
    Returns numpy array (D,) or None if no face detected.
    """
    faces = get_insightface_faces(img)
    if not faces:
        return None

    # Pick the largest face by area to reduce mismatch on group photos
    best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) if hasattr(f, 'bbox') else 0)
    base_emb = _normalize_embedding(best_face.normed_embedding)

    if not use_augment:
        return base_emb

    emb_list = [base_emb]
    for aug in augment_image(img, n=n_aug):
        faces_aug = get_insightface_faces(aug)
        if not faces_aug:
            continue
        best_aug = max(faces_aug, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) if hasattr(f, 'bbox') else 0)
        emb_list.append(_normalize_embedding(best_aug.normed_embedding))

    if len(emb_list) == 0:
        return None

    emb_mean = np.mean(np.vstack(emb_list), axis=0)
    return _normalize_embedding(emb_mean)

def get_insightface_embeddings_with_augmentation(img, n_augment=3):
    """
    More general function that supports multiple faces and augmentations.
    Returns a list of dicts like:
      [{'bbox':[xmin,ymin,w,h], 'embeddings': np.ndarray (k,D)}, ...]
    """
    faces = app.get(img)
    if not faces:
        return []

    out = []
    for f in faces:
        try:
            bbox = f.bbox.astype(int).tolist()  # [xmin, ymin, xmax, ymax]
            xmin, ymin, xmax, ymax = bbox
            w = xmax - xmin
            h = ymax - ymin
            bbox_wh = [int(xmin), int(ymin), int(w), int(h)]
        except Exception:
            bbox_wh = [0,0,0,0]

        base_emb = _normalize_embedding(f.normed_embedding)
        embeddings = [base_emb]

        # augment image and try to match augment faces by center distance
        aug_imgs = augment_image(img, n=n_augment)
        for aug in aug_imgs:
            faces_aug = app.get(aug)
            if not faces_aug:
                continue
            # pick the largest face in augmentation and append
            best_aug = max(faces_aug, key=lambda ff: (ff.bbox[2]-ff.bbox[0])*(ff.bbox[3]-ff.bbox[1]) if hasattr(ff, 'bbox') else 0)
            embeddings.append(_normalize_embedding(best_aug.normed_embedding))

        embeddings = np.vstack(embeddings)
        out.append({'bbox': bbox_wh, 'embeddings': embeddings})
    return out

def cosine_similarity(a, b):
    """
    a: (D,) normalized, b: (D,) normalized -> scalar dot product
    If not normalized, compute dot/(||a||*||b||)
    """
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    if a.ndim > 1 or b.ndim > 1:
        raise ValueError("cosine_similarity expects 1-D arrays")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def match_embedding_to_database(embedding, database, threshold=0.6):
    """
    embedding: 1-D np array (normalized)
    database: dict {label: np.ndarray (D,) } or {label: np.ndarray (k,D)} (we'll handle both)
    Returns (best_label, best_score) or ("unknown", best_score) if below threshold.
    """
    best_label = "unknown"
    best_score = -1.0
    for label, db_emb in database.items():
        db_emb = np.array(db_emb, dtype=np.float32)
        # if db_emb is 2D (k,D) average rows
        if db_emb.ndim == 2:
            db_vec = np.mean(db_emb, axis=0)
        else:
            db_vec = db_emb
        # ensure normalized
        if np.linalg.norm(db_vec) > 0:
            db_vec = db_vec / np.linalg.norm(db_vec)
        score = cosine_similarity(embedding, db_vec)
        if score > best_score:
            best_score = score
            best_label = label
    if best_score >= threshold:
        return best_label, best_score
    return "unknown", best_score