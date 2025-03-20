import cv2
import torch
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import pickle

# Load YOLO model for face detection
model = YOLO("./files/face_detector.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load stored embeddings from file
def load_face_db():
    try:
        with open("./files/face_embeddings.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading face embeddings: {e}")
        return {}

# Function to recognize face by comparing embeddings
def recognize_face(face_img, face_db):
    try:

        # Extract embedding for the new face image
        new_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        new_embedding = new_embedding / np.linalg.norm(new_embedding)  # Normalize

        best_match = None
        best_similarity = -1  # Track highest similarity

        # Compare with stored embeddings
        for person, embeddings in face_db.items():
            for stored_embedding in embeddings:
                # Ensure stored embedding is normalized
                stored_norm = np.array(stored_embedding) / np.linalg.norm(stored_embedding)
                # Compute cosine similarity
                similarity = np.dot(stored_norm, new_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person

        # Adjust similarity threshold (typical range: 0.6-0.8)
        if best_similarity >= 0.6:
            return best_match, best_similarity
        else:
            return "Unknown", None

    except Exception as e:
        print(f"Error recognizing face: {e}")
        return "Error", None

# Process Webcam Frames
cap = cv2.VideoCapture(0)  # Open webcam
# Load stored embeddings
face_db = load_face_db()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO face detection
    results = model(frame)
    faces = results[0].boxes.xyxy.cpu().numpy()  # Get detected face boxes

    for box in faces:
        x1, y1, x2, y2 = map(int, box)  # Convert to int
        face = frame[y1:y2, x1:x2]  # Crop face
        face = cv2.resize(face, (160, 160))

        # Recognize face by comparing embeddings
        try:
            best_match, confidence = recognize_face(face, face_db)

            # Display result
            if confidence is not None:
                cv2.putText(frame, f"{best_match} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f"{best_match}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        except Exception as e:
            print(f"Error processing face: {e}")

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
