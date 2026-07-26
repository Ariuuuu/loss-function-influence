import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.optim import SGD, lr_scheduler
from torch.utils.data import random_split, DataLoader, Dataset
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

writer = SummaryWriter(log_dir='runs/vector_loss/norm2')

TRAIN_NPZ = '~/train_data.npz'
TEST_NPZ  = '~/test_data.npz'

# This class transforms the data from the npz folders ready to be loaded on to the model.
class ImageNetDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, Y_orig: np.ndarray, transform=None):
        target_map = [151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 166, 168, 169, 171, 172, 173, 174, 176, 281, 282, 283, 284, 285]
        fine_map = {cls: i for i, cls in enumerate(target_map)}
        Y_orig_remapped = np.array([fine_map[cls] for cls in Y_orig], dtype='int32')
        self.X         = torch.from_numpy(X)         
        self.Y         = torch.from_numpy(Y).long()  
        self.Y_orig    = torch.from_numpy(Y_orig_remapped).long()
        self.transform = transform

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        img   = self.X[idx]
        superclass = self.Y[idx]
        label = self.Y_orig[idx]
        if self.transform:
            img = self.transform(img)
        return img, superclass, label

class Loader(DataLoader):
    def __getitem__(self, key):
        pass
# This class defines the NN used to perform the experiment
class CNN(nn.Module):
    def __init__(self, num_superclasses=2):
        super(CNN, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),

            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 4096), nn.ReLU(),  
            nn.Linear(4096, 2048),           nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.superclass = nn.Linear(2048, num_superclasses)

    def forward(self, x):  
        features = self.backbone(x)

        superclass = self.superclass(features)

        return superclass

# Training pipeline
def train(model: nn.Module, data: DataLoader, criterion, optimizer, size: int, device):

    # Variable loss_norm serves to track the loss decay over epochs, train_loss is a vector to record each of the 25 classes loss.
    # Train count serves us as a tracker of how many batches had at least one instance of a class.
    # Class accuracy tracks the number of percentage of correctly predicted data, with class_count counting the total number of instances per class.
    # Correct tracks the overall accuracy 
    loss_norm = 0.0
    train_loss = torch.zeros(25)
    train_count = torch.zeros(25)
    class_accuracy = torch.zeros(25)
    class_count = torch.zeros(25)
    correct = 0

    model.to(device)
    model.train()
    for imgs, superclasses, labels in data:
        imgs, superclasses, labels = imgs.to(device), superclasses.to(device), labels.to(device)

        super_out = model(imgs)
        loss_vector   = criterion(super_out, superclasses)

        # This variables serve as auxilariy variables to the ones previously defined, reset after each batch to ensure consistency.
        batch_loss = torch.zeros(25, device=device)
        batch_count = torch.zeros(25, device=device)
        batch_class_accuracy = torch.zeros(25, device=device)
        batch_class_count = torch.zeros(25, device=device)

        """The loop builds a mask over the batch for each of the 25 classes, identifying which instances belong to which classes. 
        After that we update the auxiliary accordingly: 
        class_loss = mean(loss of all instances from class i), batch_loss saves this value in a vector in order to than calculate a norm out of it;
        batch_count += 1 if class i is present in current batch;
        batch_class_accuracy += num of correctly classified items for class i; 
        batch_class_count += num of all instances from class i in current batch."""
        for id in labels.unique():
            mask = labels == id
            class_loss = loss_vector[mask].mean()
            batch_loss[id] += class_loss
            batch_count[id] += 1
            batch_class_accuracy[id] += super_out[mask].argmax(1).eq(superclasses[mask]).sum().item()
            batch_class_count[id] = mask.sum()

        # Auxiliary variable batch_loss is used to calculate the norm
        loss = torch.linalg.vector_norm(batch_loss, ord=np.inf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_norm += loss.item()
        train_loss += batch_loss.detach().cpu()
        train_count += batch_count.cpu()
        class_accuracy += batch_class_accuracy.cpu()
        class_count += batch_class_count.cpu()
        correct    += super_out.argmax(1).eq(superclasses).sum().item()

    return {
        "Loss": train_loss / torch.clamp(train_count, min=1),
        "Loss Norm": loss_norm / len(data),
        "Class Accuracy": class_accuracy / torch.clamp(class_count, min=1),
        "Accuracy" : correct / size
    }

# Evaluation pipeline
@torch.no_grad()
def evaluate(model: nn.Module, data: DataLoader, criterion, size: int, device,
             predictions_list: Optional[list] = None, labels_list: Optional[list] = None):
    # Variable loss_norm serves to track the overall loss decay over epochs, eval_loss is a vector to record each of the 25 classes loss.
    # Eval count serves us as a tracker of how many batches had at least one instance of a class.
    # Class accuracy tracks the number of percentage of correctly predicted data, with class_count counting the total number of instances per class.
    # Correct tracks the overall correctly classified instances
    loss_norm = 0.0
    eval_loss = torch.zeros(25)
    eval_count = torch.zeros(25)
    class_accuracy = torch.zeros(25)
    class_count = torch.zeros(25)
    correct    = 0

    model.to(device)
    model.eval()
    for imgs, superclasses, labels in data:
        imgs, superclasses, labels = imgs.to(device), superclasses.to(device), labels.to(device)
        
        super_out = model(imgs)
        loss_vector   = criterion(super_out, superclasses)

        # This variables serve as auxilariy variables to the ones previously defined, reset after each batch to ensure consistency.
        batch_loss = torch.zeros(25, device=device)
        batch_count = torch.zeros(25, device=device)
        batch_class_accuracy = torch.zeros(25, device=device)
        batch_class_count = torch.zeros(25, device=device)

        """The loop builds a mask over the batch for each of the 25 classes, identifying which instances belong to which classes. 
        After that we update the auxiliary accordingly: 
        class_loss = mean(loss of all instances from class i), batch_loss saves this value in a vector in order to than calculate a norm out of it;
        batch_count += 1 if class i is present in current batch;
        batch_class_accuracy += num of correctly classified items for class i; 
        batch_class_count += num of all instances from class i in current batch."""
        for id in labels.unique():
            mask = labels == id
            class_loss = loss_vector[mask].mean()
            batch_loss[id] += class_loss
            batch_count[id] += 1
            batch_class_accuracy[id] += super_out[mask].argmax(1).eq(superclasses[mask]).sum().item()
            batch_class_count[id] += mask.sum()

        # Auxiliary variable batch_loss is used to calculate the norm
        loss_norm += torch.linalg.vector_norm(batch_loss, ord=2).item()
        eval_loss += batch_loss.detach().cpu()
        eval_count += batch_count.cpu()
        class_accuracy += batch_class_accuracy.cpu()
        class_count += batch_class_count.cpu()
        correct += super_out.argmax(1).eq(superclasses).sum().item()

        # The empty arrays passed as arguments are used to keep track of the results and use them to generate a classification report/confussion matrix.
        if predictions_list is not None and labels_list is not None:
            _, pred = torch.max(super_out.data, 1)
            predictions_list.extend(pred.cpu().numpy())
            labels_list.extend(superclasses.cpu().numpy())

    return {
        "Loss norm": loss_norm / len(data),
        "Loss"     : eval_loss.detach() / torch.clamp(eval_count, min=1),
        "Class Accuracy": class_accuracy / torch.clamp(class_count, min=1),
        "Accuracy" : correct / size
    }

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
])

# no augmentation for val/test
test_transform = None

# Loading and unpacking train data from .npz file
train_data = np.load(TRAIN_NPZ)
X, Y, Y_orig = train_data['X'], train_data['Y'], train_data['Y_orig']

# Preprocessing the data using the custom built class.
full_dataset = ImageNetDataset(X, Y, Y_orig, transform=train_transform)

#Splitting the train data into a train set and validation set
train_size   = int(0.8 * len(full_dataset))
val_size     = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# apply test transform to validation split
val_dataset.dataset.transform = test_transform

# Preparing train and validation data
train_loader = DataLoader(train_dataset, batch_size=32,  shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=25, shuffle=False, num_workers=4)

# Loading, unpacking, preprocessing and preparing the test data
test_data    = np.load(TEST_NPZ)                      
test_dataset = ImageNetDataset(test_data['X'], test_data['Y'], test_data['Y_orig'], transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Defining model, loss criteria, optimizer, scheduler and determinig device
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = CNN(num_superclasses=2).to(device)
criterion = nn.CrossEntropyLoss(reduction='none')                      
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)

print(f"Training on: {device}")
print(f"Train: {train_size} | Val: {val_size} | Test: {len(test_dataset)}\n")

# Training loop
for epoch in range(20):
    training   = train(model, train_loader, criterion, optimizer, train_size, device)
    validation = evaluate(model, val_loader, criterion, val_size, device)
    scheduler.step(validation["Loss norm"])

    # Keeping track of training by providing some data to the human eye
    print(f"Epoch {epoch+1:02d}:\n"
          f"Training Loss Norm {training['Loss Norm']:.4f} | "
          f"Train Acc {training['Accuracy']:.2f}\n"
          f"Validation Loss Norm {validation['Loss norm']:.4f} | "
          f"Validation Acc {validation['Accuracy']:.2f}")
    
    # Using Tensorboard to record more detailed insights on training and validation during epochs
    for i in range(len(training['Loss'])):
        writer.add_scalar(f'loss/train_class_{i}', training['Loss'][i].item(), epoch)
        writer.add_scalar(f'loss/validation_class_{i}', validation['Loss'][i].item(), epoch)
        writer.add_scalar(f'accuracy/train_class{i}', training['Class Accuracy'][i].item(), epoch)
        writer.add_scalar(f'accuracy/validation_class_{i}', validation['Class Accuracy'][i].item(), epoch) 

super_predictions = []
super_labels      = []

test_metrics = evaluate(model, test_loader, criterion, len(test_dataset),
                        device, super_predictions, super_labels)

print("\n======== Test results: =========\n"
      f"Loss {test_metrics['Loss norm']:.4f} | Accuracy {test_metrics['Accuracy']:.2f}")
print("\n======== Per class test results: ========\n")
for i in range(25):
    print(f"Loss {test_metrics['Loss'][i]:.4f} | Accuracy {test_metrics['Class Accuracy']:.2f}")

print("\n======== Classification report: ========")
print(classification_report(np.array(super_labels), np.array(super_predictions),
                             target_names=['cat', 'dog']))
print(confusion_matrix(np.array(super_labels), np.array(super_predictions)))

writer.flush()
writer.close()