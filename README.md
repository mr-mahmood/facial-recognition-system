# Face Recognition System

![video](./files/output.gif)

## Overview

This project implements a face recognition system using DeepFace for facial embeddings and YOLO for face detection. It supports real-time webcam face recognition, video processing, and embedding extraction from a dataset.

## Features

- **Real-time Face Recognition**: Detects and recognizes faces from a webcam stream.
- **Video Face Recognition**: Processes a video file and annotates recognized faces.
- **Face Embedding Extraction**: Extracts facial embeddings from a dataset for future recognition.
- **Customizable Dataset**: Supports adding new identities for recognition.

## Project Structure

```bash
.
├── files
│    ├── dataset
│    │   ├── Angelina Jolie
│    │   ├── Brad Pitt
│    │   ├── Denzel Washington
│    │   ├── ... (other identities)
│    ├── face_detector.pt  # Pretrained YOLO model
│    ├── face_embeddings.pkl  # Stored face embeddings
│    ├── input.mp4  # Sample video for processing
│    ├── output.mp4  # Processed output video
├─ extract_facial_embeddings.py
├─ video_result.py
├─ webcam_result.py
├─ README.md

```
## Dependencies

Ensure you have the following dependencies installed:

```bash
pip install opencv-python torch numpy deepface ultralytics tqdm pickle5
```
## Usage

### 1. Extract Face Embeddings
Run the following script to extract embeddings from images stored in `files/dataset`:
```bash
python extract_embeddings.py
```

### 2. Real-time Face Recognition
Run the following command to start real-time face recognition using a webcam:
```bash
python webcam_result.py
```

### 3. Process Video for Face Recognition
To process a video file and annotate recognized faces:
```bash
python video_result.py
```

## How It Works
1. **Face Detection**: Uses YOLO to detect faces in an image or video frame.
2. **Face Embedding Extraction**: DeepFace extracts facial embeddings.
3. **Face Matching**: Uses cosine similarity to compare new faces against stored embeddings.
4. **Recognition & Annotation**: Matches are displayed along with confidence scores.

## Customizing the Dataset
To add new identities, place their images in `files/dataset/{person_name}` and re-run `extract_embeddings.py`.

## Notes
- Ensure `face_detector.pt` is placed in `./files/`.
- Adjust similarity threshold in `recognize_face()` if needed.
- The system supports CUDA acceleration if available.
- this implementation is not very fast and take its time to run and i will try to make it faster and real time
- it is not very accurate and make some mistake as it is clear in video at top of readme file so i can make it better for sure

## License
This project is for educational purposes only. Use at your own discretion.
