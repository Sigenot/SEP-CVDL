import cv2
import numpy as np
import torch
from model_arch import EmotionCNN
import preprocess

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
        processed_face = preprocess(face_img)

        # predict emotions
        with torch.no_grad():
            emotion_prediction = model(processed_face)
        emotion_label = emotion_labels[torch.argmax(emotion_prediction)]

        # draw a rectangle around face and label predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # show frame
    cv2.imshow('Emotion Recognition', frame)

    # end the loop when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# clear resources
cap.release()
cv2.destroyAllWindows()
