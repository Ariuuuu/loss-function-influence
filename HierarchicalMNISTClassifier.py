import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


class MNIST(datasets.MNIST):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        superclass = label % 2 
        return image, superclass, label 
    
class HierarchicalNN(nn.Module):
    def __init__(self):
        super(HierarchicalNN, self).__init__()
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
        self.digit_head = nn.Linear(128, 10)

        self.register_buffer("even_mask", torch.tensor([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float))
        self.register_buffer("odd_mask", torch.tensor([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float))
    
    def forward(self, x):
        features = self.backbone(x)

        super_logits = self.odd_even_head(features)
        super_probs = F.softmax(super_logits, dim=1)

        p_even = super_probs[:, 0:1]
        p_odd = super_probs[:, 1:2]

        soft_mask = p_even * self.even_mask + p_odd * self.odd_mask

        digit_logits = self.digit_head(features)
        masked_logits = digit_logits + (soft_mask + 1e-10).log()
        
        return super_logits, masked_logits

def train(model, data, criterion, optimizer, device, size):
    train_loss = 0
    correct_super, correct_digit = 0, 0

    model = model.to(device)
    model.train()
    for img, super, digit in data:
        img, super, digit = img.to(device), super.to(device), digit.to(device)

        super_pred, digit_pred = model(img)

        loss_digit = criterion(digit_pred, digit)
        # loss_super = criterion(super_pred, super)
        loss = loss_digit # + 0.1*loss_super

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct_super += super_pred.argmax(1).eq(super).sum().item()
        correct_digit += digit_pred.argmax(1).eq(digit).sum().item()
        train_loss += loss.item()

    return {
        "loss": train_loss / len(data), 
        "super acc": correct_super * 100 / size,
        "digit acc": correct_digit * 100 / size
    }

@torch.no_grad()
def evaluation(model, data, criterion, device, size,
               super_prediction=None, super_label=None, digit_prediction=None, digit_label=None):
    total_loss = 0
    correct_super, correct_digit = 0, 0

    model = model.to(device)
    model.eval()
    for img, super, digit in data:
        img, super, digit = img.to(device), super.to(device), digit.to(device)

        super_pred, digit_pred = model(img)

        loss_digit = criterion(digit_pred, digit)
        # loss_super = criterion(super_pred, super)
        loss = loss_digit # + 0.1*loss_super

        correct_super += super_pred.argmax(1).eq(super).sum().item()
        correct_digit += digit_pred.argmax(1).eq(digit).sum().item()
        total_loss += loss.item()

        if (all(v is not None for v in (super_prediction, super_label, digit_prediction, digit_label))):
            _, super_pred = torch.max(super_pred.data, 1)
            _, digit_pred = torch.max(digit_pred.data, 1)
            super_prediction.extend(super_pred.cpu().numpy())
            super_label.extend(super.cpu().numpy())
            digit_prediction.extend(digit_pred.cpu().numpy())
            digit_label.extend(digit.cpu().numpy())
    
    return {
        "loss": total_loss / len(data),
        "super acc": correct_super * 100 / size,
        "digit acc": correct_digit * 100 / size
    }

model = HierarchicalNN()
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


for epoch in range(20):
    training = train(model, train_loader, criterion, optimizer, device, train_size)
    validitation = evaluation(model, val_loader, criterion, device, val_size)
    scheduler.step(validitation["loss"])

    print(f"Epoch {epoch+1}\n"
          f"Training results:\n"
          f"Loss {training['loss']:.5f} | Superclass accuracy {training['super acc']:.3f}% | Digit accuracy {training['digit acc']:.3f}%\n"
          f"Validation metrics:\n"
          f"Loss {validitation['loss']:.5f} | Superclass accuracy {validitation['super acc']:.3f}% | Digit accuracy {validitation['digit acc']:.3f}%")

super_predictions = []
super_labels = []
digit_predictions = []
digit_labels = []
test_metrics = evaluation(model, test_loader, criterion, device, len(test_dataset), super_predictions, super_labels, digit_predictions, digit_labels)

print("======== Test results: =========\n"
      f"Loss {test_metrics['loss']:.5f} | Superclass accuracy {test_metrics['super acc']:.3f}% | Digit accuracy {test_metrics['digit acc']:.3f}%")

print("\n ======== Superclass classification report: ========")
print(classification_report(np.array(super_labels), np.array(super_predictions)))
print(confusion_matrix(np.array(super_labels), np.array(super_predictions)))
print("\n ======== Digit classification report: ========")
print(classification_report(np.array(digit_labels), np.array(digit_predictions)))
print(confusion_matrix(np.array(digit_labels), np.array(digit_predictions)))