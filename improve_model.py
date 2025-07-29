import torch
import torch.nn as nn
import torch.optim as optim
from model_arch import EmotionCNN
from dataloader import train_loader, test_loader

# check if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# init model
model = EmotionCNN().to(device)

# define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# further train model
epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0 
    for images, labels in train_loader:
        # move data to gpu
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # calc avg loss
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    # validate model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            # move data to gpu
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100 * correct / total}%")

# save model
torch.save(model.state_dict(), 'further_trained_fer_model.pth')