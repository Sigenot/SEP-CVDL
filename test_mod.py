from model_arch import EmotionCNN
from dataloader import test_loader
import torch
from collections import OrderedDict

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

# initialize vars for accuracy calculation
correct = 0
total = 0

# deactivate calculation of gradients for prediction
with torch.no_grad():
    for images, labels in test_loader:
        outputs = loaded_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total}%")