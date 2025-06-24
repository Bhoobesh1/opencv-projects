import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Step 1: Load known faces
known_face_encodings = []
known_face_names = []

faces_path = "faces"

print("[INFO] Loading known faces...")

for file_name in os.listdir(faces_path):
    if file_name.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(faces_path, file_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(file_name)[0]
            known_face_names.append(name)
            print(f"[INFO] Loaded encoding for: {name}")
        else:
            print(f"[WARNING] No face found in {file_name} - Skipping.")

# Step 2: Initialize attendance record
attendance_marked = set()

def mark_attendance(name):
    if name not in attendance_marked:
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        with open("attendance.csv", "a") as f:
            f.write(f"{name},{timestamp}\n")
        attendance_marked.add(name)
        print(f"[✔️] Marked attendance for {name} at {timestamp}")

# Step 3: Start webcam
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

print("[INFO] Webcam started. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to read from webcam.")
        break

    # Resize frame for faster face processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)

    if face_locations:
        try:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        except Exception as e:
            print(f"[ERROR] Face encoding failed: {e}")
            continue

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    mark_attendance(name)

            # Scale face locations back to original frame size
            top, right, bottom, left = face_location
            top *= 4; right *= 4; bottom *= 4; left *= 4

            # Draw box and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        print("[INFO] No face detected in frame.")

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting.")
        break

video_capture.release()
cv2.destroyAllWindows()
