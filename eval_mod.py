from model_arch import EmotionCNN
from dataloader import eval_loader
import torch
from collections import OrderedDict
import torch.nn.functional as F

# emotion labels
emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

# create instance of the model
loaded_model = EmotionCNN()

# load the saved model
state_dict = torch.load('fer_model.pth', map_location=torch.device('cpu'))#print(state_dict) # debug

# adjust keys in dictionary
new_state_dict = OrderedDict()
for key, value in state_dict.items():
    if key.startswith('layer'):
        parts = key.split('.')
        if len(parts) == 2:
            new_key = f"{parts[0]}.0.{parts[1]}"
        else: 
            new_key = key
    else:
        new_key = key
    new_state_dict[new_key] = value

# load the state of the model into the instance
loaded_model.load_state_dict(new_state_dict)

# set model to evaluation mode
loaded_model.eval()
'''
# initialize vars for accuracy calculation
correct = 0
total = 0

# deactivate calculation of gradients for prediction
with torch.no_grad():
    for images, labels in eval_loader:
        outputs = loaded_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
'''
# initialize a variable to collect probabilities
correct_predictions = torch.zeros(len(emotions))
total_predictions = torch.zeros(len(emotions))

emotion_probabilities = torch.zeros(len(emotions))


# deactivate calculation of gradients for prediction of emotion probabilities
with torch.no_grad():
    total = 0
    for images, _ in eval_loader:
        outputs = loaded_model(images)
        probabilities = F.softmax(outputs, dim=1)
        emotion_probabilities += torch.sum(probabilities, dim=0)
        total += images.size(0)

# deactivate calculations of gradients for prediction of probabilities (each emotion)
with torch.no_grad():
    for images, labels in eval_loader:
        outputs = loaded_model(images)
        _, predicted = torch.max(outputs.data, 1)

        # count correct predictions for each emotion
        for label, pred in zip(labels, predicted):
            if label == pred:
                correct_predictions[label] += 1
            total_predictions[label] += 1

# calc avg probabilities
emotion_probabilities /= total

# print avg probabilities for each emotion
print("Average probabilites for each emotion")
for label, prob in zip(emotions, emotion_probabilities):
    print(f"{label}: {prob.item() * 100:.2f}%")

# print the procentual share of each correctly guessed emotion
print()
print("Correctly guessed emotions")
for label, correct, total in zip(emotions, correct_predictions, total_predictions):
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"{label}: {accuracy:.2f}%")
    else:
        print(f"{label}: N/A")

#print(f"Test Accuracy: {100 * correct / total}%")