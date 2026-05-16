import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.batchNorm2 = nn.BatchNorm2d(64)
        self.dropout_conv = nn.Dropout2d(0.25)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64*7*7, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout_fc = nn.Dropout(0.5)

    def forward(self, x):
        x = self.batchNorm1(self.conv1(x))
        x = self.pool(F.relu(x))
        x = self.dropout_conv(x)
        x = self.batchNorm2(self.conv2(x))
        x = self.pool(F.relu(x))
        x = self.dropout_conv(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.dropout_fc(x)
        x = self.fc3(x)

        return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.01, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

train_transformer = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, (0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))
])

test_transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))
])

# Retrieving the full MNIST train dataset and splitting it in a train set and validation set (ratio 80:20)
full_size = datasets.MNIST(root='/home/eriseldo/Desktop/Bachelorarbeit', train=True, transform=train_transformer)
train_size = (int) (0.8*len(full_size))
val_size = len(full_size) - train_size
train_dataset, val_dataset = random_split(full_size, [train_size, val_size])

# Loading each of the validation and train set, set the validation transformer to test_transformer
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataset.dataset.transform = test_transformer
val_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False, num_workers=4)

# Loading the test set 
test_dataset = datasets.MNIST(root='/home/eriseldo/Desktop/Bachelorarbeit', train=False, transform=test_transformer)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, num_workers=4)

# Moving the model, data, labels to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

# Metrics to interpret results 
train_losses = []
val_losses = []
val_accuracy = []

for epoch in range(30):
    train_loss = 0     # accumulating train losses of all batches
    
    # Setting the model on train mode and train using the train set
    net.train()
    for image, label in train_loader:
        image, label = image.to(device), label.to(device)

        output = net(image)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()  # accumulate
    
    val_loss = 0 # accumulating validation losses of all batches
    correct = 0  # accumulating correctly classified instances

    # Setting the model on evaluation mode and using the validation set to evaluate after each train epoch
    net.eval()
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)    
            output = net(images)
            loss = criterion(output, labels)
            val_loss += loss.item()
            correct += output.argmax(1).eq(labels).sum().item()

    avg_train_loss = train_loss/len(train_loader)                     # average the train loss over all batches
    avg_val_loss = val_loss/len(val_loader)                           # average the validation loss over all batches
    val_acc = (correct)*100/val_size              # calculating an intermediate accuracy on validation set

    scheduler.step(avg_val_loss)
    train_losses.append(avg_train_loss)   
    val_losses.append(avg_val_loss)         
    val_accuracy.append(val_acc)    
    print(f"Epoch {epoch+1} finished: Train loss {avg_train_loss:.5f} | Val loss {avg_val_loss:.5f} | Val accuracy {val_acc:.3f}")

print("")
print(train_losses)
print("")
print(val_losses)
print("")
print(val_accuracy)
print("")

all_preds = []
all_labels = []

net.eval()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

print(classification_report(all_labels, all_preds))
print(confusion_matrix(all_labels, all_preds))
# print(roc_auc_score(all_labels, all_preds))
# print(roc_curve(all_labels, all_preds))