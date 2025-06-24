import cv2
import os
import numpy as np

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

# Train using your saved images
recognizer, label_map = train_model("faces")
