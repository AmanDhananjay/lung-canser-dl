import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = 128  # Resize images

def load_images(folder):
    images, labels = [], []
    for label in os.listdir(folder):
        class_path = os.path.join(folder, label)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0  # Normalize
            images.append(img)
            labels.append(0 if label == "normal" else 1)
    return np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1), np.array(labels)

X, y = load_images("dataset")
np.save("X.npy", X)
np.save("y.npy", y)

print("Dataset Preprocessing Complete!")
