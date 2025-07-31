import cv2
import numpy as np
#from tensorflow.keras.models import load_model
import torch
from model_arch import EmotionCNN
from preprocess import preprocess_img

# load pretrained model
model = EmotionCNN()
state_dict = torch.load('new_mod.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

def process_video(input_path, output_path):
    # Öffnen des Videos
    cap = cv2.VideoCapture(input_path)

    # Überprüfen, ob das Video erfolgreich geöffnet wurde
    if not cap.isOpened():
        print("Fehler beim Öffnen des Videos")
        return

    # Holen der Videodetails: Breite, Höhe und FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Erstellen eines VideoWriter-Objekts, um das neue Video zu speichern
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Vorverarbeitung des Frames für das Modell
        processed_frame = preprocess_img(frame)

        # Emotion klassifizieren
        emotion = classify_emotion(model, processed_frame)

        # Saliency Map erstellen
        saliency_map = create_saliency_map(model, processed_frame)

        # Saliency Map und Emotion auf dem Frame anzeigen
        frame_with_info = overlay_info(frame, emotion, saliency_map)

        # Frame in das Ausgabevideo schreiben
        out.write(frame_with_info)

    # Freigabe der Ressourcen
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def classify_emotion(model, frame):
    # Emotion klassifizieren
    predictions = model.predict(np.expand_dims(frame, axis=0))
    emotion_label = np.argmax(predictions)
    return emotion_label

def create_saliency_map(model, frame):
    # Saliency Map erstellen
    # Hier kannst du eine Methode verwenden, um die Saliency Map zu erstellen
    # Zum Beispiel mit Grad-CAM oder einer anderen Methode
    saliency_map = np.zeros_like(frame)
    return saliency_map

def overlay_info(frame, emotion, saliency_map):
    # Emotion und Saliency Map auf dem Frame ^anzeigen
    cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    frame = cv2.addWeighted(frame, 1, saliency_map, 0.5, 0)
    return frame

# Beispielaufruf der Funktion
process_video('input_video.mp4', 'output_video.mp4')