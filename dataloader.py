import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from load_images import load_img_and_lbl

# get data from images
train_faces, train_labels = load_img_and_lbl('train')
test_faces, test_labels = load_img_and_lbl('test')

# convert to pytorch-tensors
faces_tensor = torch.tensor(train_faces, dtype=torch.float32)
labels_tensor = torch.tensor(np.argmax(train_labels, axis=1), dtype=torch.long)

# custom dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
    
# create dataset and dataloader
train_dataset = CustomDataset(faces_tensor, labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# repeat for test data
test_faces_tensor = torch.tensor(test_faces, dtype=torch.float32)
test_labels_tensor = torch.tensor(np.argmax(test_labels, axis=1), dtype=torch.long)
test_dataset = CustomDataset(test_faces_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)