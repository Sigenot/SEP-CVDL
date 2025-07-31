import os
import cv2
import numpy as np

import csv
import os

# dataset paths to evaluation data
dataset_path_1 = 'eval_affectnet'
dataset_path_2 = 'eval_ck'

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

def load_img(dataset_type):
    faces = []
    labels = []
    image_paths = []
    dataset_paths = [dataset_path_1, dataset_path_2]

    # iterate the file
    for dataset_path in dataset_paths:
        for emotion_index, emotion in enumerate(emotions):
            emotion_path = os.path.join(dataset_path, dataset_type, emotion)
            if not os.path.exists(emotion_path):
                continue
            for img_name in os.listdir(emotion_path):
                img_path = os.path.join(emotion_path, img_name)

                # load image and convert to grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                img = cv2.resize(img, (64, 64))

                faces.append(img)
                labels.append(emotion_index)
                image_paths.append(img_path)

    # convert lists to numpy arrays
    faces = np.asarray(faces)
    labels = np.asarray(labels)

    # normalize images
    faces = faces / 255.0

    # convert labels to one-hot encoding
    num_classes = len(emotions)
    labels_one_hot = np.zeros((len(labels), num_classes))
    labels_one_hot[np.arange(len(labels)), labels] = 1

    # reshape for model
    faces = faces.reshape(faces.shape[0], 64, 64, 1)

    return faces, labels_one_hot