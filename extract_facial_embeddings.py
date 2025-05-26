import os
import cv2
import faiss
import pickle
import numpy as np
from datetime import datetime
from deepface import DeepFace

# ==== CONFIG ====
EMBEDDING_DIM = 128  # DeepFace Facenet default is 128-dim
EMBEDDINGS_PATH = "./files/faiss/face_index.faiss"
LABELS_PATH = "./files/faiss/labels.pkl"
DATASET_DIR = "./files/dataset"
PERSON_NAME = "poya"  # ← Change this before running
NUM_IMAGES = 5
BOX_SIZE = 170
CROP_MARGIN = 10
# ===============

# Create directories
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
person_folder = os.path.join(DATASET_DIR, PERSON_NAME)
os.makedirs(person_folder, exist_ok=True)

# Load or create FAISS index and labels
if os.path.exists(EMBEDDINGS_PATH) and os.path.exists(LABELS_PATH):
    index = faiss.read_index(EMBEDDINGS_PATH)
    with open(LABELS_PATH, "rb") as f:
        labels = pickle.load(f)
    print("[INFO] Loaded existing FAISS index and labels.")
else:
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    labels = []
    print("[INFO] Initialized new FAISS index and label list.")

def draw_green_box(frame, box_size=BOX_SIZE):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    x1, y1 = cx - box_size // 2, cy - box_size // 2
    x2, y2 = cx + box_size // 2, cy + box_size // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def get_face_box_coords(frame, box_size=BOX_SIZE, margin=CROP_MARGIN):
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    x1 = max(cx - box_size // 2 + margin, 0)
    y1 = max(cy - box_size // 2 + margin, 0)
    x2 = min(cx + box_size // 2 - margin, w)
    y2 = min(cy + box_size // 2 - margin, h)
    return x1, y1, x2, y2

def normalize(embedding):
    return embedding / np.linalg.norm(embedding)

# === IMAGE CAPTURE PHASE ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot access webcam.")

print(f"[INFO] Align your face inside the green box. Press SPACE to capture ({NUM_IMAGES} total).")

image_paths = []
captured = 0

while captured < NUM_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame.")
        continue

    display_frame = frame.copy()
    draw_green_box(display_frame)
    cv2.imshow("Capture Face", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        x1, y1, x2, y2 = get_face_box_coords(frame)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            print("[WARN] Empty face crop, try again.")
            continue

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_name = f"{PERSON_NAME}_{timestamp}.jpg"
        img_path = os.path.join(person_folder, img_name)
        cv2.imwrite(img_path, face_crop)
        image_paths.append(img_path)
        captured += 1
        print(f"[INFO] Captured {captured}/{NUM_IMAGES}: {img_path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# === EMBEDDING PHASE ===
print("[INFO] Extracting embeddings and adding to FAISS index...")

for img_path in image_paths:
    try:
        emb_obj = DeepFace.represent(img_path=img_path, model_name="Facenet", enforce_detection=False)[0]
        emb = normalize(np.array(emb_obj["embedding"]).astype(np.float32))

        index.add(np.expand_dims(emb, axis=0))
        labels.append(PERSON_NAME)
        print(f"[SUCCESS] Embedded and added: {img_path}")
    except Exception as e:
        print(f"[ERROR] Failed to embed {img_path}: {e}")

# Save FAISS index and labels
faiss.write_index(index, EMBEDDINGS_PATH)
with open(LABELS_PATH, "wb") as f:
    pickle.dump(labels, f)

print("✅ All images and embeddings saved successfully.")
