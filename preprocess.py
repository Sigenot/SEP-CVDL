import torch
import numpy as np
import cv2

def preprocess_img(frame, target_size=(64, 64)):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, target_size)
    gray = gray / 255.0
    gray = np.expand_dims(gray, axis=-1)
    gray = np.expand_dims(gray, axis=0)
    gray_tensor = torch.tensor(gray, dtype=torch.float32).permute(0, 3, 1, 2)
    return gray_tensor