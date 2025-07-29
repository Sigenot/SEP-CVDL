import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from load_images import load_img_and_lbl
from load_eval_data import load_img

# get data from images
train_faces, train_labels = load_img_and_lbl('train')
test_faces, test_labels = load_img_and_lbl('test')

eval_faces, eval_lables = load_img('test')



# convert to pytorch-tensors
faces_tensor = torch.tensor(train_faces, dtype=torch.float32).permute(0, 3, 1, 2)
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
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# repeat for test data
test_faces_tensor = torch.tensor(test_faces, dtype=torch.float32).permute(0, 3, 1, 2)
test_labels_tensor = torch.tensor(np.argmax(test_labels, axis=1), dtype=torch.long)
test_dataset = CustomDataset(test_faces_tensor, test_labels_tensor)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# repeat for eval data
eval_faces_tensor = torch.tensor(eval_faces, dtype=torch.float32).permute(0,3,1,2)
eval_labels_tensor = torch.tensor(np.argmax(eval_lables, axis=1), dtype=torch.long)
eval_dataset = CustomDataset(eval_faces_tensor, eval_labels_tensor)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)