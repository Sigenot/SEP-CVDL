import os
from keras.utils import to_categorical
import numpy as np
import cv2

# path to folders
dataset_path = 'dataset'

# emotions and respective folders
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

def load_img_and_lbl(dataset_type):
    faces = []
    labels = []
    # iterating through file
    for emotion_index, emotion in enumerate(emotions):
        emotion_path = os.path.join(dataset_path, dataset_type, emotion)
        for img_name in os.listdir(emotion_path):
            img_path = os.path.join(emotion_path, img_name)
            
            # load image and convert to grayscale (designed redundantly)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64,64))
            
            faces.append(img)
            labels.append(emotion_index)
            
    # convert lists to numpyArrays
    faces = np.asarray(faces)
    labels = np.asarray(labels)

    # normalizing images
    faces = faces / 255.0

    # convert labels
    labels = to_categorical(labels, num_classes=len(emotions))
    
    # reshape for modell
    faces = faces.reshape(faces.shape[0], 64, 64, 1)
    
    return faces, labels