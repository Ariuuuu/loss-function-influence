import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from typing import Literal, Optional
import wandb

class MNIST(datasets.MNIST):
    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        superclass = label % 2
        return image, superclass
    
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            nn.Flatten(),
            nn.Linear(64*5*5, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.odd_even_head = nn.Linear(128, 2)

    def forward(self, x):
        features = self.backbone(x)
        output = self.odd_even_head(features)

        return output 
    
def train(model: nn.Module, data: DataLoader, criterion, optimizer: torch.optim, size: int, device = Literal["cuda", "cpu"]):
    train_loss = 0
    correct = 0

    model.to(device)
    model.train()
    for img, label in data:
        img, label = img.to(device), label.to(device)

        output = model(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += output.argmax(1).eq(label).sum().item()

    return {
        "Loss" : train_loss / len(data),
        "Accuracy" : correct * 100 / size
    }

@torch.no_grad()
def evaluate(model: nn.Module, data: DataLoader, criterion, size: int, device = Literal["cuda", "cpu"], predictions: Optional[list] = None, labels: Optional[list] = None):
    total_loss = 0
    correct = 0

    model.to(device)
    model.eval()
    for img, label in data:
        img, label = img.to(device), label.to(device)

        output = model(img)
        loss = criterion(output, label)

        total_loss += loss.item()
        correct += output.argmax(1).eq(label).sum().item()

        if (predictions is not None and labels is not None):
            _, pred = torch.max(output.data, 1)
            predictions.extend(pred.cpu().numpy())
            labels.extend(label.cpu().numpy())

    return {
        "Loss" : total_loss / len(data),
        "Accuracy" : correct * 100 / size
    }

model = cnn()
criterion = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr = 0.01, momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)

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
full_size = MNIST(root='/home/eriseldo/Desktop/Bachelorarbeit', train=True, transform=test_transformer)
train_size = (int) (0.8*len(full_size))
val_size = len(full_size) - train_size
train_dataset, val_dataset = random_split(full_size, [train_size, val_size])

# Loading each of the validation and train set, set the validation transformer to test_transformer
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_dataset.dataset.transform = test_transformer
val_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False, num_workers=4)

# Loading the test set 
test_dataset = MNIST(root='/home/eriseldo/Desktop/Bachelorarbeit', train=False, transform=test_transformer)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, num_workers=4)

# Moving the model, data, labels to gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializing wandb tracker to monitor training
wandb.init(project='odd-even',
           config={"epochs": 20, "lr": 0.01, "scheduler": 1, "augmentation": 1}) 

for epoch in range(20):
    training = train(model, train_loader, criterion, optimizer,  train_size, device)
    validation = evaluate(model, val_loader, criterion, val_size, device)
    scheduler.step(validation["Loss"])

    wandb.log({"loss/train": training['Loss'], "loss/validation": validation['Loss'], 
               "accuracy/train": training['Accuracy'],'accuracy/validation': validation['Accuracy']}, step=epoch)
    

super_predictions = []
super_labels = []

test_metrics = evaluate(model, test_loader, criterion, len(test_dataset), device, super_predictions, super_labels)

print("\n======== Test results: =========\n"
      f"Loss {test_metrics['Loss']:.5f} | Accuracy {test_metrics['Accuracy']:.3f}%")

print("\n======== Superclass classification report: ========")
print(classification_report(np.array(super_labels), np.array(super_predictions)))
print(confusion_matrix(np.array(super_labels), np.array(super_predictions)))
wandb.finish()