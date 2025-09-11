import os
import json
import sqlite3
from datetime import datetime, date
from typing import Dict, Tuple

import cv2
import numpy as np
import pandas as pd


MODELS_DIR = os.path.join(os.getcwd(), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "face_lbph.xml")
LABELS_PATH = os.path.join(MODELS_DIR, "labels.json")
DB_PATH = os.path.join(os.getcwd(), "attendance.db")
EXPORTS_DIR = os.path.join(os.getcwd(), "exports")


def ensure_setup() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EXPORTS_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                date TEXT NOT NULL,
                in_time TEXT,
                out_time TEXT,
                status TEXT NOT NULL DEFAULT 'partial',
                created_at TEXT NOT NULL
            )
            """
        )
        con.commit()


def load_model_and_labels() -> Tuple[cv2.face_LBPHFaceRecognizer, Dict[int, str]]:
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(LABELS_PATH):
        raise RuntimeError("Model or labels not found. Train first by running collect_and_train.py")

    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        raise RuntimeError("cv2.face not found. Install 'opencv-contrib-python'.")

    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        id_to_name = {int(k): v for k, v in meta.get("id_to_name", {}).items()}

    return recognizer, id_to_name


def get_face_detector() -> cv2.CascadeClassifier:
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar cascade for face detection.")
    return face_cascade


def log_attendance(student_name: str, attendance_type: str = "in") -> str:
    """
    Log attendance for a student (in-time or out-time)
    Returns status message for display
    """
    now = datetime.utcnow()
    today = now.date().isoformat()
    timestamp = now.time().strftime("%H:%M:%S")
    
    with sqlite3.connect(DB_PATH) as con:
        # Check if student already has attendance record for today
        cursor = con.execute(
            "SELECT id, in_time, out_time, status FROM attendance WHERE student_name = ? AND date = ?",
            (student_name, today)
        )
        record = cursor.fetchone()
        
        if record:
            record_id, in_time, out_time, current_status = record
            
            if attendance_type == "in":
                if in_time:
                    return f"{student_name} already marked IN at {in_time}"
                else:
                    # Update with in-time
                    con.execute(
                        "UPDATE attendance SET in_time = ?, status = CASE WHEN out_time IS NOT NULL THEN 'present' ELSE 'partial' END WHERE id = ?",
                        (timestamp, record_id)
                    )
                    con.commit()
                    return f"{student_name} marked IN at {timestamp}"
            else:  # out-time
                if out_time:
                    return f"{student_name} already marked OUT at {out_time}"
                elif not in_time:
                    return f"{student_name} must mark IN first before marking OUT"
                else:
                    # Update with out-time
                    con.execute(
                        "UPDATE attendance SET out_time = ?, status = 'present' WHERE id = ?",
                        (timestamp, record_id)
                    )
                    con.commit()
                    return f"{student_name} marked OUT at {timestamp}"
        else:
            # New record for today
            if attendance_type == "in":
                con.execute(
                    "INSERT INTO attendance (student_name, date, in_time, status, created_at) VALUES (?, ?, ?, 'partial', ?)",
                    (student_name, today, timestamp, now.isoformat())
                )
                con.commit()
                return f"{student_name} marked IN at {timestamp}"
            else:
                return f"{student_name} must mark IN first before marking OUT"


def export_today_to_excel() -> str:
    today_str = date.today().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            """
            SELECT 
                student_name,
                in_time,
                out_time,
                status,
                CASE 
                    WHEN status = 'present' THEN 'Full Attendance'
                    WHEN status = 'partial' THEN 'Partial Attendance (Missing OUT)'
                    ELSE 'Absent'
                END as attendance_status
            FROM attendance 
            WHERE date = ?
            ORDER BY student_name
            """,
            con,
            params=(today_str,),
        )
    
    if df.empty:
        filepath = os.path.join(EXPORTS_DIR, f"attendance_{today_str}_EMPTY.xlsx")
        pd.DataFrame({"info": ["No attendance records for today."]}).to_excel(filepath, index=False)
        return filepath

    # Add summary statistics
    summary_data = {
        'Total Students': [len(df)],
        'Full Attendance': [len(df[df['status'] == 'present'])],
        'Partial Attendance': [len(df[df['status'] == 'partial'])],
        'Absent': [len(df[df['status'] == 'absent'])]
    }
    summary_df = pd.DataFrame(summary_data)
    
    filepath = os.path.join(EXPORTS_DIR, f"attendance_{today_str}.xlsx")
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Attendance Details', index=False)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    return filepath


def recognize_loop(confidence_threshold: float = 70.0) -> None:
    recognizer, id_to_name = load_model_and_labels()
    face_cascade = get_face_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Webcam not accessible.")

    print("[INFO] === Face Recognition Attendance System ===")
    print("[INFO] Press 'q' to quit")
    print("[INFO] Press 'e' to export today's attendance to Excel")
    print("[INFO] Press 'i' to mark IN-time")
    print("[INFO] Press 'o' to mark OUT-time")
    print("[INFO] Press 'a' to add new student to dataset")
    print("[INFO] Press 'r' to retrain model with new data")
    print("[INFO] ==========================================")

    last_attendance_time = {}  # Track last attendance time to avoid spam
    attendance_cooldown = 5  # seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

        # Draw instructions on frame
        cv2.putText(frame, "Press 'i' for IN-time, 'o' for OUT-time", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'a' to add student, 'e' to export", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (200, 200))

            label_id, confidence = recognizer.predict(face_resized)
            # Lower confidence is better in LBPH; convert to score for display
            display_text = "Unknown Face"
            color = (0, 0, 255)
            status_text = "Not registered - Press 'a' to add"

            if label_id in id_to_name and confidence < confidence_threshold:
                student_name = id_to_name[label_id]
                display_text = f"{student_name} ({confidence:.1f})"
                color = (0, 255, 0)
                status_text = "Registered - Press 'i' or 'o'"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, display_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, status_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Recognition & Attendance", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('e'):
            path = export_today_to_excel()
            print(f"[INFO] Exported today's attendance to: {path}")
        elif key == ord('a'):
            add_new_student()
        elif key == ord('r'):
            retrain_model()
        elif key == ord('i') or key == ord('o'):
            # Process attendance for detected faces
            if faces is not None and len(faces) > 0:
                # Use the first detected face
                (x, y, w, h) = faces[0]
                face_roi = gray[y:y + h, x:x + w]
                face_resized = cv2.resize(face_roi, (200, 200))
                
                label_id, confidence = recognizer.predict(face_resized)
                
                if label_id in id_to_name and confidence < confidence_threshold:
                    student_name = id_to_name[label_id]
                    attendance_type = "in" if key == ord('i') else "out"
                    
                    # Check cooldown
                    now = datetime.utcnow()
                    if student_name in last_attendance_time:
                        if (now - last_attendance_time[student_name]).seconds < attendance_cooldown:
                            print(f"[INFO] Please wait {attendance_cooldown} seconds before next attendance for {student_name}")
                            continue
                    
                    message = log_attendance(student_name, attendance_type)
                    print(f"[INFO] {message}")
                    last_attendance_time[student_name] = now
                else:
                    print("[INFO] No registered face detected. Press 'a' to add new student.")
            else:
                print("[INFO] No face detected. Please position your face in front of the camera.")

    cap.release()
    cv2.destroyAllWindows()


def add_new_student():
    """Add a new student to the dataset during runtime"""
    print("\n=== Add New Student ===")
    student_name = input("Enter Student Name/ID: ").strip()
    if not student_name:
        print("[INFO] No name provided. Cancelled.")
        return
    
    try:
        samples_text = input("Number of samples to capture [default 30]: ").strip()
        samples = int(samples_text) if samples_text else 30
    except ValueError:
        samples = 30
    
    # Import the capture function from collect_and_train
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from collect_and_train import capture_faces_for_student
        capture_faces_for_student(student_name, samples)
        print(f"[INFO] Successfully captured {samples} samples for {student_name}")
        print("[INFO] Press 'r' to retrain the model with new data")
    except Exception as e:
        print(f"[ERROR] Failed to capture faces: {e}")


def retrain_model():
    """Retrain the model with updated dataset"""
    print("\n=== Retraining Model ===")
    try:
        from collect_and_train import train_and_save_model
        train_and_save_model()
        print("[INFO] Model retrained successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to retrain model: {e}")


def main():
    ensure_setup()
    recognize_loop()


if __name__ == "__main__":
    main()


