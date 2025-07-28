import cv2
import numpy as np
import torch
import torch.nn.functional as F
import time
from model_arch import EmotionCNN
from preprocess import preprocess_img

# load pretrained model
model = EmotionCNN()
state_dict = torch.load('fer_model.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# emotion labels
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

# load haar-cascade classifier für facial recognition
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

#start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # recognize face in frame
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        processed_face = preprocess_img(face_img)

        # predict emotions
        with torch.no_grad():
            emotion_prediction = model(processed_face)
            probabilities = F.softmax(emotion_prediction, dim=1)[0] * 100

        #emotion_label = emotion_labels[torch.argmax(emotion_prediction)]

        # draw a rectangle around face and label predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # show probability of all emotions
        text_x = x + w + 100
        y_offset = 20

        # show probability of all emotions
        for i, (label, prob) in enumerate(zip(emotion_labels, probabilities)):
            text = f"{label}: {prob:.2f}%"
            cv2.putText(frame, text, (x, y + y_offset * i -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # show frame
    cv2.imshow('Emotion Recognition', frame)

    # reduce refresh rate
    #time.sleep(0.1)

    # end the loop when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clear resources
cap.release()
cv2.destroyAllWindows()
