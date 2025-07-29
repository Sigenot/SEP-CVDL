from load_images import load_img_and_lbl
from torchvision import transforms
from grad_cam import grad_cam
from torch.utils.data import Dataset
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
from model_arch import EmotionCNN
import torch
from collections import OrderedDict

random.seed(12345)

emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']

loaded_model = EmotionCNN()

# load weights of model
state_dict = torch.load('fer_model.pth', map_location=torch.device('cpu'))

# adjust keys in dict
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

# load state of model to instance
loaded_model.load_state_dict(new_state_dict)

# set mod to eval mode
loaded_model.eval()

# choose target layer for grad cam
target_layer = loaded_model.layer4[2]

# load images and labels
faces, labels = load_img_and_lbl('test')

class FacesDataset(Dataset):
    def __init__(self, faces, labels, transform=None):
        self.faces = faces
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.faces)
    
    def __getitem__(self, index):
        face = self.faces[index]
        label = self.labels[index]

        if self.transform:
            face = self.transform(face)

        return face, label

# transform to tensor
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    #transforms.Lambda(lambda x: x.repeat(3,1,1)) # convert grayscale to rgb
])

# create dataset
dataset = FacesDataset(faces, labels, transform=transform)



# select 5 random images from test dataset
selected_indices = random.sample(range(len(dataset)), 5)

# function to save results
def save_results(original_image, gradcam, combined_image, label, filename):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title(f"Original Image\nLabel: {emotions[label]}")
    plt.imshow(original_image.squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Grad-CAM")
    plt.imshow(gradcam, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Combined Image")
    plt.imshow(combined_image)
    plt.axis('off')

    plt.savefig(filename)
    plt.close()

# use grad cam on each selected image
for index in selected_indices:
    image, label = dataset[index]
    image_tensor = image.unsqueeze(0)

    # calculate grad cam
    gradcam = grad_cam(loaded_model, image_tensor, target_layer)

    # visualize and save result
    heatmap = cv2.applyColorMap(np.uint8(255 * gradcam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (64, 64))
    heatmap = heatmap / 255.0

    # combine grayscale image with heatmap
    gray_image = image.squeeze().numpy()
    gray_image_3d = np.stack([gray_image] * 3, axis=-1)

    combined_img = heatmap * 0.5 + gray_image_3d * 0.5

    # make sure label index is int
    label_index = int(label.argmax())

    save_results(image, heatmap, combined_img, label_index, f'gradcam_{index}.png')