import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

np.random.seed(42)
n = 2 # The norm to be later used
writer = SummaryWriter(log_dir=f'runs/vector_loss/norm{n}')

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

#This class is used to sample the batches used during training in order to ensure balance
class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, drop_last=False):
        self.labels = np.array(labels)
        self.groups = np.unique(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.per_group = self.batch_size // len(self.groups)
        if self.per_group == 0:
            raise ValueError("batch size must be >=  number of groups")

        self.group_indices = {
            g: np.where(self.labels == g)[0] for g in self.groups
        }

    def __iter__(self):
        pools = {g: np.random.permutation(idx) for g, idx in self.group_indices.items()}
        pointers = {g: 0 for g in self.groups}

        while True:
            batch = []

            active_groups = [g for g in self.groups if pointers[g] < len(self.group_indices[g])]

            if not active_groups:
                return

            num_active = len(active_groups)
            base = self.batch_size // num_active
            remainder = self.batch_size % num_active

            for i, g in enumerate(active_groups):
                alloc = base + (1 if i < remainder else 0)
                start = pointers[g]
                end = start + alloc

                if end > len(self.group_indices[g]):
                    end = len(self.group_indices[g])
                batch.extend(pools[g][start:end])
                pointers[g] = end

            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        min_group_size = min(len(idx) for idx in self.group_indices.values())
        return min_group_size // self.per_group
        
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

            nn.Conv2d(64, 128, 3),
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
def train(model: nn.Module, data: DataLoader, criterion, optimizer, size: int, norm: int, epoch: int, device):
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
    step = 1

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

            if epoch<=20:
                writer.add_scalar(f"batch_loss/train_class{id}", class_loss, epoch*1000+step)
                writer.add_scalar(f"batch_accuracy/train_class{id}", batch_class_accuracy[id], epoch*1000+step)

        # Auxiliary variable batch_loss is used to calculate the norm
        loss = torch.linalg.vector_norm(batch_loss, ord=norm)
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
def evaluate(model: nn.Module, data: DataLoader, criterion, size: int, norm: int, epoch: int, device,
             predictions_list: Optional[list] = None, labels_list: Optional[list] = None, test: bool = False):
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
    step = 1

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

            if epoch<=20:
                if test:
                    writer.add_scalar(f"batch_loss/test_class{id}", class_loss, epoch*100+step)
                    writer.add_scalar(f"batch_accuracy/test_class{id}", batch_class_accuracy[id] / batch_class_count[id], epoch*100+step)
                else:
                    writer.add_scalar(f"batch_loss/val_class{id}", class_loss, epoch*100+step)
                    writer.add_scalar(f"batch_accuracy/val_class{id}", batch_class_accuracy[id] / batch_class_count[id], epoch*100+step)

        # Auxiliary variable batch_loss is used to calculate the norm
        loss_norm += torch.linalg.vector_norm(batch_loss, ord=norm).item()
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
labels = full_dataset.Y_orig

#Splitting the train data into a train set and validation set where all classes share equal proportions
train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.1, stratify=labels, random_state=42)
train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
val_dataset = torch.utils.data.Subset(full_dataset, val_idx)
train_size = len(train_dataset)
val_size = len(val_dataset)

# Defining a sampler for train data
orig_train_data = train_dataset.dataset
orig_train_indices = train_dataset.indices 
training_sampler = BalancedBatchSampler(orig_train_data.Y_orig[orig_train_indices], 250)
# Preparing train data
train_loader = DataLoader(train_dataset,  batch_sampler=training_sampler,  num_workers=4)

# Defining a sampler for validation data
orig_val_data = val_dataset.dataset
orig_val_indices = val_dataset.indices
val_sampler = BalancedBatchSampler(orig_val_data.Y_orig[orig_val_indices], 250)
# Preparing validation data
val_loader   = DataLoader(val_dataset,  batch_sampler=val_sampler, num_workers=4)

# Loading, unpacking, preprocessing and preparing the test data
test_data    = np.load(TEST_NPZ)                      
test_dataset = ImageNetDataset(test_data['X'], test_data['Y'], test_data['Y_orig'], transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=250, shuffle=False, num_workers=4)

# Defining model, loss criteria, optimizer, scheduler and determinig device
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = CNN(num_superclasses=2).to(device)
criterion = nn.CrossEntropyLoss(reduction='none')                      
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)

print(f"Training on: {device}")
print(f"Train: {train_size} | Val: {val_size} | Test: {len(test_dataset)}\n")

# Training loop
for epoch in range(100):
    training   = train(model, train_loader, criterion, optimizer, train_size, n, epoch+1, device)
    validation = evaluate(model, val_loader, criterion, val_size, n, epoch+1, device)
    scheduler.step(validation["Loss norm"])
    test = evaluate(model, test_loader, criterion, len(test_dataset), n, epoch+1, device)

    # Keeping track of training by providing some data to the human eye
    print(f"Epoch {epoch+1:02d}:\n"
          f"Training Loss Norm {training['Loss Norm']:.4f} | "
          f"Train Acc {training['Accuracy']:.2f}\n"
          f"Validation Loss Norm {validation['Loss norm']:.4f} | "
          f"Validation Acc {validation['Accuracy']:.2f}\n"
          f"Test Loss Norm {test['Loss norm']:.4f} | "
          f"Test Acc {test['Accuracy']:.2f}")
    
    # Using Tensorboard to record more detailed insights on training and validation during epochs
    for i in range(len(training['Loss'])):
        writer.add_scalar(f'loss/train_class_{i}', training['Loss'][i].item(), epoch)
        writer.add_scalar(f'loss/validation_class_{i}', validation['Loss'][i].item(), epoch)
        writer.add_scalar(f'accuracy/train_class{i}', training['Class Accuracy'][i].item(), epoch)
        writer.add_scalar(f'accuracy/validation_class_{i}', validation['Class Accuracy'][i].item(), epoch) 
        writer.add_scalar(f'test_accuracy/test_class{i}', test['Class Accuracy'][i].item(), epoch)
        writer.add_scalar(f'test_ovr_acc/test', test['Accuracy'], epoch)

super_predictions = []
super_labels      = []

test_metrics = evaluate(model, test_loader, criterion, len(test_dataset), 
                        n, epoch+1, device, super_predictions, super_labels)

print("\n======== Classification report: ========")
print(classification_report(np.array(super_labels), np.array(super_predictions),
                             target_names=['cat', 'dog']))
print(confusion_matrix(np.array(super_labels), np.array(super_predictions)))

writer.flush()
writer.close()