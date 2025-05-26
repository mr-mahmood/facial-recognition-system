import os
import cv2
import faiss
import pickle
import torch
import numpy as np
import time
from deepface import DeepFace
from ultralytics import YOLO

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==== CONFIG ====
INDEX_PATH = "./files/faiss/face_index.faiss"
LABELS_PATH = "./files/faiss/labels.pkl"
DETECTION_MODEL_PATH = "./files/face_detector.pt"  # Trained YOLO model path
EMBEDDING_DIM = 128  # DeepFace Facenet model produces 128-dim embeddings
SIMILARITY_THRESHOLD = 0.6
# ================

os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLO model
print("[INFO] Loading YOLO model...")
yolo_model = YOLO(DETECTION_MODEL_PATH).to(device)
#yolo_model = YOLO(DETECTION_MODEL_PATH)

# Load or create FAISS index
def load_faiss_and_labels():
    if os.path.exists(INDEX_PATH) and os.path.exists(LABELS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(LABELS_PATH, "rb") as f:
            labels = pickle.load(f)
        print("[INFO] FAISS index and labels loaded.")
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIM)
        labels = []
        print("[INFO] Created new FAISS index and label list.")
    return index, labels

# Normalize embedding
def normalize(embedding):
    return embedding / np.linalg.norm(embedding)

# Recognize face using DeepFace + FAISS
def recognize_face(face_img, index, labels, timings):
    try:
        start_embed = time.time()
        result = DeepFace.represent(img_path=face_img, model_name='Facenet', enforce_detection=False)[0]
        embed_time = time.time() - start_embed
        timings['embedding'].append(embed_time)

        start_norm = time.time()
        embedding = normalize(np.array(result["embedding"]).astype(np.float32))
        norm_time = time.time() - start_norm
        timings['normalize'].append(norm_time)

        start_search = time.time()
        D, I = index.search(np.expand_dims(embedding, axis=0), k=1)
        search_time = time.time() - start_search
        timings['faiss_search'].append(search_time)

        similarity = D[0][0]
        idx = I[0][0]

        if similarity >= SIMILARITY_THRESHOLD:
            return labels[idx], similarity
        else:
            return "Unknown", None

    except Exception as e:
        print(f"[ERROR] Failed to recognize face: {e}")
        return "Error", None

# ==== MAIN ====
index, labels = load_faiss_and_labels()
cap = cv2.VideoCapture(0)
timings = {
    'frame_capture': [], 'yolo_detection': [],
    'embedding': [], 'normalize': [], 'faiss_search': []
}

if not cap.isOpened():
    print("[FATAL] Webcam not accessible.")
    exit(1)

ORIG_H, ORIG_W = None, None
input_size = 640

print("[INFO] Starting video stream. Press 'q' to quit.")
while cap.isOpened():
    start_frame = time.time()
    ret, frame = cap.read()
    frame_capture_time = time.time() - start_frame
    timings['frame_capture'].append(frame_capture_time)

    if not ret:
        break

    if ORIG_H is None or ORIG_W is None:
        ORIG_H, ORIG_W = frame.shape[:2]

    resized_frame = cv2.resize(frame, (input_size, input_size))

    start_yolo = time.time()
    results = yolo_model(resized_frame, verbose=True)
    yolo_time = time.time() - start_yolo
    timings['yolo_detection'].append(yolo_time)

    boxes = results[0].boxes.xyxy.cpu().numpy()
    scale_x = ORIG_W / input_size
    scale_y = ORIG_H / input_size

    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = int(x1 * scale_x)
        x2 = int(x2 * scale_x)
        y1 = int(y1 * scale_y)
        y2 = int(y2 * scale_y)

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        # Save face temporarily to disk (required by DeepFace)
        temp_path = "temp_face.jpg"
        cv2.imwrite(temp_path, face)

        name, confidence = recognize_face(temp_path, index, labels, timings)

        if confidence is not None:
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# === TIMING REPORT ===
print("\n=== Average timings (seconds) ===")
for k, v in timings.items():
    if v:
        print(f"{k}: {sum(v)/len(v):.4f} (avg over {len(v)} runs)")
