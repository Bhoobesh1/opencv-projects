import cv2
import os
import numpy as np

# Step 1: Train the model
def train_model(data_dir):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_id = 0
    name_label_map = {}

    for name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, name)
        if not os.path.isdir(folder_path):
            continue

        name_label_map[label_id] = name

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(label_id)

        label_id += 1

    recognizer.train(faces, np.array(labels))
    return recognizer, name_label_map

# Train using your saved images (inside "faces" folder)
recognizer, label_map = train_model("faces")

# Step 2: Real-time recognition
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        label, confidence = recognizer.predict(roi)

        if confidence < 70:
            name = label_map[label]
        else:
            name = "Person"

        cv2.putText(frame, f"{name}  confidence:{round(confidence,2)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
