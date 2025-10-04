# mediapipe_detection.py
import cv2
import mediapipe as mp

def detect_faces_mediapipe(img, min_detection_confidence=0.5):
    """
    Returns a list of face dicts:
      [
        {"bbox": [xmin, ymin, width, height], "score": <float>},
        ...
      ]
    Coordinates are in pixel space (ints).
    """
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_detection_confidence) as face_detection:
        results = face_detection.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.detections:
            return []

        h, w, _ = img.shape
        faces = []
        for det in results.detections:
            rbox = det.location_data.relative_bounding_box
            xmin = int(max(0, rbox.xmin * w))
            ymin = int(max(0, rbox.ymin * h))
            width = int(max(0, rbox.width * w))
            height = int(max(0, rbox.height * h))
            score = float(det.score[0]) if det.score else 0.0
            faces.append({
                "bbox": [xmin, ymin, width, height],
                "score": score
            })
        return faces