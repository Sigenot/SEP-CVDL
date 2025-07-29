from model_arch import EmotionCNN
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import train_loader, test_loader, eval_loader

# check if gpu is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# initialize model
model = EmotionCNN().to(device)

# define optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# variables for csv 
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
eval_losses = []
eval_accuracies = []

timeout = 10
best_value = 0
counter = 0

# train the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # calculate avg loss
    #avg_loss = running_loss / len(train_loader)
    #print(f"Epoch {epoch+1}, Loss: {avg_loss}")



    # test model
    model.eval()

    running_loss_test = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            # move data to gpu
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss_test += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    test_loss = running_loss_test / len(test_loader)
    test_accuracy = test_correct / test_total
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    

    # get eval-method into train, extra dataset
    model.eval()
    running_loss_eval = 0.0
    eval_correct = 0
    eval_total = 0

    with torch.no_grad():
        for images, labels in eval_loader:
            # move data to gpu
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss_eval += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            eval_total += labels.size(0)
            eval_correct += (predicted == labels).sum().item()
    eval_loss = running_loss_eval / len(eval_loader)
    eval_accuracy = eval_correct / eval_total
    eval_losses.append(eval_loss)
    eval_accuracies.append(eval_accuracy)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss}, Train Accuracy: {train_accuracy}, Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Eval Loss: {eval_loss}, Eval Accuracy: {eval_accuracy}%")

# method to stop improvement declines

    if eval_accuracy > best_value:
        best_value = eval_accuracy
        counter = 0
        torch.save(model.state_dict(), 'new_mod.pth')
    else:
        counter += 1
        print(f"No improvement for {counter} epochs.%")

    if counter > timeout:
        print("Lacking improvement: STOP")
        break






