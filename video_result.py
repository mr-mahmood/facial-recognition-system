import cv2
import torch
import numpy as np
from deepface import DeepFace
from ultralytics import YOLO
import pickle


model = YOLO("./files/face_detector.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
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

        new_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]
        new_embedding = new_embedding / np.linalg.norm(new_embedding)  # Normalize

        best_match = None
        best_similarity = -1


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

input_video_path = "./files/input.mp4" 
output_video_path = "./files/output.mp4"

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"Error: Could not open video {input_video_path}.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height))

# Load stored embeddings
face_db = load_face_db()


frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break


    faces = model(frame, verbose=False)[0]

    for detect_face in faces.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detect_face
        if score > 0.5:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (160, 160))  # Resize for embedding extraction


            try:
                best_match, confidence = recognize_face(face, face_db)

                if confidence is not None:
                    cv2.putText(frame, f"{best_match} ({confidence:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, f"{best_match}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            except Exception as e:
                print(f"Error processing face: {e}")

    out.write(frame)

    frame_count += 1
    print(f"Processed frame {frame_count}/{total_frames}")

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing complete. Output saved to {output_video_path}")