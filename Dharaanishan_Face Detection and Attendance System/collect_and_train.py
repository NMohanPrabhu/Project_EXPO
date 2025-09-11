import os
import json
import cv2
import numpy as np
from datetime import datetime


DATASET_DIR = os.path.join(os.getcwd(), "dataset")
MODELS_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "face_lbph.xml")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")


def ensure_directories() -> None:
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def get_face_detector() -> cv2.CascadeClassifier:
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")
    return face_cascade


def capture_faces_for_student(student_name: str, samples_per_student: int = 30) -> None:
    student_dir = os.path.join(DATASET_DIR, student_name)
    os.makedirs(student_dir, exist_ok=True)

    face_cascade = get_face_detector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible. Ensure a camera is connected and not in use.")

    print(f"[INFO] Capturing {samples_per_student} face samples for '{student_name}'. Press 'q' to abort.")
    captured = 0

    try:
        while captured < samples_per_student:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to read frame from camera.")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

            for (x, y, w, h) in faces:
                face_roi = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face_roi, (200, 200))
                img_path = os.path.join(student_dir, f"{student_name}_{captured + 1:03d}.jpg")
                cv2.imwrite(img_path, face_resized)
                captured += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{student_name} {captured}/{samples_per_student}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if captured >= samples_per_student:
                    break

            cv2.imshow("Capture Faces", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Capture aborted by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"[INFO] Collected {captured} images for {student_name} at '{student_dir}'.")


def build_training_data():
    images = []
    labels = []
    name_to_id = {}
    current_id = 0

    if not os.path.isdir(DATASET_DIR):
        raise RuntimeError(f"Dataset directory not found at {DATASET_DIR}. Capture data first.")

    for student_name in sorted(os.listdir(DATASET_DIR)):
        student_dir = os.path.join(DATASET_DIR, student_name)
        if not os.path.isdir(student_dir):
            continue

        if student_name not in name_to_id:
            name_to_id[student_name] = current_id
            current_id += 1

        label_id = name_to_id[student_name]
        for file in sorted(os.listdir(student_dir)):
            if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(student_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue
            # Ensure consistent size
            img = cv2.resize(img, (200, 200))
            images.append(img)
            labels.append(label_id)

    if not images:
        raise RuntimeError("No training images found. Please capture data first.")

    return images, np.array(labels, dtype=np.int32), name_to_id


def train_and_save_model():
    print("[INFO] Preparing training data...")
    images, labels, name_to_id = build_training_data()

    # Inverse map for clarity and potential future use
    id_to_name = {v: k for k, v in name_to_id.items()}

    print("[INFO] Training LBPH face recognizer...")
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    except AttributeError:
        raise RuntimeError(
            "cv2.face not found. Install 'opencv-contrib-python' (pip install opencv-contrib-python)."
        )

    recognizer.train(images, labels)

    os.makedirs(MODELS_DIR, exist_ok=True)
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, 'w', encoding='utf-8') as f:
        json.dump({"name_to_id": name_to_id, "id_to_name": id_to_name, "trained_at": datetime.utcnow().isoformat()}, f, indent=2)

    print(f"[INFO] Model saved to: {MODEL_PATH}")
    print(f"[INFO] Labels saved to: {LABELS_PATH}")


def add_student_interactive():
    """Interactive function to add a single student"""
    ensure_directories()
    
    print("=== Add New Student ===")
    print("Instructions:\n - Look at the camera in good lighting.\n - Keep a neutral face; slight angles help variability.\n - Press 'q' during capture to abort the current student.")
    
    student_name = input("Enter Student Name/ID: ").strip()
    if not student_name:
        print("[INFO] No name provided. Cancelled.")
        return False
    
    try:
        samples_text = input("Number of samples to capture [default 30]: ").strip()
        samples = int(samples_text) if samples_text else 30
    except ValueError:
        samples = 30

    capture_faces_for_student(student_name, samples)
    return True


def main():
    ensure_directories()

    print("=== Student Data Collection & Training ===")
    print("Instructions:\n - Look at the camera in good lighting.\n - Keep a neutral face; slight angles help variability.\n - Press 'q' during capture to abort the current student.")

    while True:
        student_name = input("Enter Student Name/ID (or press Enter to stop adding): ").strip()
        if not student_name:
            break
        try:
            samples_text = input("Number of samples to capture [default 30]: ").strip()
            samples = int(samples_text) if samples_text else 30
        except ValueError:
            samples = 30

        capture_faces_for_student(student_name, samples)

    print("[INFO] Starting training...")
    train_and_save_model()
    print("[DONE] Training complete. You can now run recognition and attendance.")


if __name__ == "__main__":
    main()


