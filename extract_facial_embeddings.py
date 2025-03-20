import os
import pickle
import numpy as np
from deepface import DeepFace
from tqdm import tqdm

# Store embeddings
face_db = {}

# Load dataset and extract embeddings
def load_dataset(directory):
    global face_db
    face_db = {}
    
    sub_dirs = [sub_dir for sub_dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, sub_dir))]
    
    for sub_dir in tqdm(sub_dirs, desc="Processing Folders"):
        sub_dir_path = os.path.join(directory, sub_dir)
        images = os.listdir(sub_dir_path)
        
        for image in tqdm(images, desc=f"Processing {sub_dir}", leave=False):
            img_path = os.path.join(sub_dir_path, image)
            try:
                # Extract face embedding
                embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                
                # Store in dictionary
                if sub_dir not in face_db:
                    face_db[sub_dir] = []
                face_db[sub_dir].append(embedding)
                
            except Exception as e:
                pass  # Skipping error prints

# Load train images and extract embeddings
load_dataset('./files/dataset')

# Save embeddings for future use
with open("./files/face_embeddings.pkl", "wb") as f:
    pickle.dump(face_db, f)

print("🔹 Face embeddings saved successfully!")