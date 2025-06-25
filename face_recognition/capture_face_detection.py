import cv2
import os

name = input("enter the name of the person")

# Create directory to store face images
folder = f"faces/{name}"
os.makedirs(folder, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(frame, 1.1, 5)

    for (x, y, w, h) in faces:
        roi = frame[y:y+h, x:x+w]    
        count += 1 
        cv2.imwrite(f"{folder}/{count}.jpg", roi)
        cv2.putText(frame, f"Saved: {count}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow("Capturing Face Images", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 10:
        break

cap.release()
cv2.destroyAllWindows()

