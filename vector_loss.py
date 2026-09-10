import random
import json
from pathlib import Path
import torchvision.transforms.v2 as v2
import torch
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Optional

TRAIN_NPZ = 'dataset/train_data.npz'
TEST_NPZ  = 'dataset/test_data_normalized.npz'


def set_seed(seed: int):
    """Seeds every RNG that affects this pipeline."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

# This class is used to sample the batches used during training in order to ensure balance
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


class pNormScheduler():
    """This class schedules the norm order which is used to calculate the loss.
    NOTE: It is being used by the sweep below, comment the line initializng it 
    and  calling pNormscheduler.step() for norm_value to be held fixed for the 
    full run."""

    def __init__(self, norm, factor=0.9, threshold=0.05, patience=5, min_norm=2.0, mode: str = 'min'):
        self.norm = norm
        self.factor = factor
        self.threshold = threshold
        self.patience = patience
        self.min_norm = min_norm
        self.mode = mode
        self.streak = 0
        self.best = np.inf if self.mode == 'min' else -np.inf

    def step(self, loss):
        improved = (loss < self.best - self.threshold) if self.mode == 'min' \
                   else (loss > self.best + self.threshold)

        if improved:
            self.best = loss
            self.streak = 0
        else:
            self.streak += 1

        if self.streak >= self.patience:
            self.norm = max(self.norm * self.factor, self.min_norm)
            self.streak = 0

        return self.norm


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
            nn.Linear(128 * 14 * 14, 12544), nn.ReLU(),
            nn.Linear(12544, 6272),           nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.superclass = nn.Linear(6272, num_superclasses)

    def forward(self, x):
        features = self.backbone(x)
        superclass = self.superclass(features)
        return superclass

# Training pipeline
def train(model: nn.Module, data: DataLoader, criterion, optimizer, size: int, norm: int, epoch: int, device, writer: SummaryWriter, augmenter):
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
        
        imgs=augmenter(imgs)
        
        super_out = model(imgs)
        loss_vector = criterion(super_out, superclasses)
        
        batch_loss = torch.zeros(25, device=device)
        batch_count = torch.zeros(25, device=device)
        batch_class_accuracy = torch.zeros(25, device=device)
        batch_class_count = torch.zeros(25, device=device)

        for id in labels.unique():
            mask = labels == id
            class_loss = loss_vector[mask].mean()
            batch_loss[id] += class_loss
            batch_count[id] += 1
            batch_class_accuracy[id] += super_out[mask].argmax(1).eq(superclasses[mask]).sum().item()
            batch_class_count[id] = mask.sum()

        loss = torch.linalg.vector_norm(batch_loss, ord=norm)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_norm += loss.item()
        train_loss += batch_loss.detach().cpu()
        train_count += batch_count.cpu()
        class_accuracy += batch_class_accuracy.cpu()
        class_count += batch_class_count.cpu()
        correct += super_out.argmax(1).eq(superclasses).sum().item()

    return {
        "Loss": train_loss / torch.clamp(train_count, min=1),
        "Loss Norm": loss_norm / len(data),
        "Class Accuracy": class_accuracy / torch.clamp(class_count, min=1),
        "Accuracy": correct / size
    }


# Evaluation pipeline
@torch.no_grad()
def evaluate(model: nn.Module, data: DataLoader, criterion, size: int, norm: int, epoch: int, device, writer: SummaryWriter,
             predictions_list: Optional[list] = None, labels_list: Optional[list] = None):
    loss_norm = 0.0
    eval_loss = torch.zeros(25)
    eval_count = torch.zeros(25)
    class_accuracy = torch.zeros(25)
    class_count = torch.zeros(25)
    correct = 0
    
    super_correct = torch.zeros(2, device=device)
    super_total = torch.zeros(2, device=device)

    model.to(device)
    model.eval()
    for imgs, superclasses, labels in data:
        imgs, superclasses, labels = imgs.to(device), superclasses.to(device), labels.to(device)

        super_out = model(imgs)
        loss_vector = criterion(super_out, superclasses)
        correct_mask = super_out.argmax(1).eq(superclasses)
        for sc in [0, 1]:
            sc_mask = (superclasses == sc)
            super_correct[sc] += (correct_mask & sc_mask).sum()
            super_total[sc]   += sc_mask.sum()

        batch_loss = torch.zeros(25, device=device)
        batch_count = torch.zeros(25, device=device)
        batch_class_accuracy = torch.zeros(25, device=device)
        batch_class_count = torch.zeros(25, device=device)

        for id in labels.unique():
            mask = labels == id
            class_loss = loss_vector[mask].mean()
            batch_loss[id] += class_loss
            batch_count[id] += 1
            batch_class_accuracy[id] += super_out[mask].argmax(1).eq(superclasses[mask]).sum().item()
            batch_class_count[id] += mask.sum()

        loss_norm += torch.linalg.vector_norm(batch_loss, ord=norm).item()
        eval_loss += batch_loss.detach().cpu()
        eval_count += batch_count.cpu()
        class_accuracy += batch_class_accuracy.cpu()
        class_count += batch_class_count.cpu()
        correct += super_out.argmax(1).eq(superclasses).sum().item()

        if predictions_list is not None and labels_list is not None:
            _, pred = torch.max(super_out.data, 1)
            predictions_list.extend(pred.cpu().numpy())
            labels_list.extend(superclasses.cpu().numpy())
            
        cat_acc = (super_correct[0] / torch.clamp(super_total[0], min=1)).item()
        dog_acc = (super_correct[1] / torch.clamp(super_total[1], min=1)).item()

    return {
        "Loss norm": loss_norm / len(data),
        "Loss": eval_loss.detach() / torch.clamp(eval_count, min=1),
        "Class Accuracy": class_accuracy / torch.clamp(class_count, min=1),
        "Accuracy": correct / size,
        "Cat Accuracy": cat_acc,
        "Dog Accuracy": dog_acc
    }


def load_raw_data(train_npz: str, test_npz: str):
    """Loads the .npz files once so every (norm, seed) run reuses the same
    arrays instead of hitting disk again on every run."""
    train_data = np.load(train_npz)
    test_data  = np.load(test_npz)
    return (train_data['X'], train_data['Y'], train_data['Y_orig'],
            test_data['X'],  test_data['Y'],  test_data['Y_orig'])


def run_experiment(norm_value: int, seed: int,
                    X_train, Y_train, Y_orig_train,
                    X_test, Y_test, Y_orig_test,
                    epochs: int = 100, device=None, num_workers: int = 0, log_batches: bool = False):
    """Runs one full train/val/test cycle for a fixed norm value and seed.
    The norm scheduler is being used so comment out lines 363 and 376 if you 
    want the norm value to remain static over all epochs.

    num_workers defaults to 0 on purpose: with several of these running
    concurrently on a single-CPU-core node (one process per GPU), spawning
    DataLoader worker processes just adds fork/IPC overhead with nowhere to
    actually run in parallel. Data loading happens synchronously in the main
    process instead, and augmentation (see gpu_augment) happens on-device."""

    set_seed(seed)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter(log_dir=f'runs/vector_loss/norm{norm_value}_seed{seed}')
    gpu_transforms = v2.Compose([
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=10),
    v2.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ])

    # No CPU-side transform anymore - augmentation happens batched, on-device,
    # inside train() via gpu_augment(). This also means the validation split
    # below no longer accidentally gets training-style augmentation applied
    # to it (the old code built both train_dataset and val_dataset as Subsets
    # of the same full_dataset, which shared the train_transform).
    full_dataset = ImageNetDataset(X_train, Y_train, Y_orig_train, transform=None)
    labels = full_dataset.Y_orig

    # The split is also re-drawn per seed (random_state=seed), so "seed"
    # controls the whole pipeline end to end (split + init + shuffling).
    # If you'd rather keep one fixed train/val split and only vary model
    # init/training stochasticity, hardcode random_state=42 here instead.
    train_idx, val_idx = train_test_split(np.arange(len(labels)), test_size=0.1, stratify=labels, random_state=seed)
    train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
    val_dataset   = torch.utils.data.Subset(full_dataset, val_idx)
    train_size = len(train_dataset)
    val_size   = len(val_dataset)

    orig_train_data = train_dataset.dataset
    orig_train_indices = train_dataset.indices
    training_sampler = BalancedBatchSampler(orig_train_data.Y_orig[orig_train_indices], 250)
    train_loader = DataLoader(train_dataset, batch_sampler=training_sampler, num_workers=num_workers)

    orig_val_data = val_dataset.dataset
    orig_val_indices = val_dataset.indices
    val_sampler = BalancedBatchSampler(orig_val_data.Y_orig[orig_val_indices], 250)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=num_workers)

    test_dataset = ImageNetDataset(X_test, Y_test, Y_orig_test, transform=None)
    test_loader  = DataLoader(test_dataset, batch_size=250, shuffle=False, num_workers=num_workers)

    model     = CNN(num_superclasses=2).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3)

    # Comment out this line, if you do not want the norm value to be reduced upon detecting plateaus
    norm_scheduler = pNormScheduler(norm_value, factor=0.7, threshold=0.01, patience=10, min_norm=2)

    print(f"\n=== norm={norm_value} | seed={seed} | device={device} ===")
    print(f"Train: {train_size} | Val: {val_size} | Test: {len(test_dataset)}\n")

    for epoch in range(epochs):
        training   = train(model, train_loader, criterion, optimizer, train_size, norm_value, epoch+1, device, writer, gpu_transforms)
        validation = evaluate(model, val_loader, criterion, val_size, norm_value, epoch+1, device, writer)
        scheduler.step(validation["Loss norm"])
        test = evaluate(model, test_loader, criterion, len(test_dataset), norm_value, epoch+1, device, writer)

        # Comment out this line, if you do not want the norm value to be reduced upon detecting plateaus
        # If you want the norm value to be reduced, than check the signal you want to be used
        norm_value = norm_scheduler.step(validation['Loss norm'])

        for i in range(len(training['Loss'])):
            writer.add_scalar(f'loss/train_class_{i}', training['Loss'][i].item(), epoch)
            writer.add_scalar(f'loss/validation_class_{i}', validation['Loss'][i].item(), epoch)
            writer.add_scalar(f'accuracy/train_class{i}', training['Class Accuracy'][i].item(), epoch)
            writer.add_scalar(f'accuracy/validation_class_{i}', validation['Class Accuracy'][i].item(), epoch)
            writer.add_scalar(f'test_accuracy/test_class{i}', test['Class Accuracy'][i].item(), epoch)
        writer.add_scalar('test_ovr_acc/test', test['Accuracy'], epoch)
        writer.add_scalar('test_ovr_acc/dog', test['Cat Accuracy'], epoch)
        writer.add_scalar('test_ovr_acc/cat', test['Dog Accuracy'], epoch)

    super_predictions, super_labels = [], []
    test_metrics = evaluate(model, test_loader, criterion, len(test_dataset),
                             norm_value, epochs, device, writer,
                             super_predictions, super_labels)

    report_dict = classification_report(np.array(super_labels), np.array(super_predictions),
                                         target_names=['cat', 'dog'], output_dict=True)
    cm = confusion_matrix(np.array(super_labels), np.array(super_predictions))

    print(f"\n======== norm={norm_value} seed={seed} - Classification report: ========")
    print(classification_report(np.array(super_labels), np.array(super_predictions), target_names=['cat', 'dog']))
    print(cm)

    writer.flush()
    writer.close()

    return {
        'norm': norm_value,
        'seed': seed,
        'test_accuracy': test_metrics['Accuracy'],
        'test_loss_norm': test_metrics['Loss norm'],
        'classification_report': report_dict,
        'confusion_matrix': cm.tolist(),
    }

_worker_gpu_id = None
_worker_data = None

def _init_worker(gpu_queue, train_npz, test_npz):
    global _worker_gpu_id, _worker_data
    _worker_gpu_id = gpu_queue.get()
    torch.cuda.set_device(_worker_gpu_id)
    _worker_data = load_raw_data(train_npz, test_npz)


def _run_one(args):
    norm_value, seed, epochs, num_workers, log_batches = args
    X_train, Y_train, Y_orig_train, X_test, Y_test, Y_orig_test = _worker_data
    device = torch.device(f'cuda:{_worker_gpu_id}')

    print(f"[GPU {_worker_gpu_id}] starting norm={norm_value} seed={seed}")
    result = run_experiment(norm_value, seed,
                             X_train, Y_train, Y_orig_train,
                             X_test, Y_test, Y_orig_test,
                             epochs=epochs, device=device,
                             num_workers=num_workers, log_batches=log_batches)
    print(f"[GPU {_worker_gpu_id}] finished norm={norm_value} seed={seed}")
    return result


def save_results(results, out_path):
    """Merges `results` into whatever's already at out_path instead of
    overwriting the file. Entries are keyed by (norm, seed)."""
    out_path = Path(out_path)
 
    existing = []
    if out_path.exists():
        try:
            with open(out_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            existing = []  # corrupt/empty file - don't let that lose new results
 
    merged = {(r['norm'], r['seed']): r for r in existing}
    merged.update({(r['norm'], r['seed']): r for r in results})
    merged_list = sorted(merged.values(), key=lambda r: (r['norm'], r['seed']))
 
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(merged_list, f, indent=2)
 
    return merged_list
 
 
def run_sweep_parallel(norm_values, seeds, train_npz=TRAIN_NPZ, test_npz=TEST_NPZ,
                        epochs=100, num_gpus=4, num_workers=0, log_batches=False,
                        results_path='runs/vector_loss/sweep_results.json'):
    """Runs every (norm, seed) combination across num_gpus GPUs in parallel,
    via torch.multiprocessing with the 'spawn' start method. 
    Results are merged into results_path (see save_results) rather than
    overwriting it. Returns the full merged list (past runs + this call's), 
    not just this call's new results."""

    ctx = mp.get_context('spawn')
    gpu_queue = ctx.Queue()
    for gpu_id in range(num_gpus):
        gpu_queue.put(gpu_id)
 
    tasks = [
        (norm_value, seed, epochs, num_workers, log_batches)
        for norm_value in norm_values
        for seed in seeds
    ]
 
    with ctx.Pool(processes=num_gpus, initializer=_init_worker, initargs=(gpu_queue, train_npz, test_npz)) as pool:
        results = pool.map(_run_one, tasks)
 
    return save_results(results, results_path)


if __name__ == '__main__':
    # ---- sweep configuration: edit these ----
    norm_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    seeds       = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, np.inf]
    epochs      = 100
    # ------------------------------------------

    X_train, Y_train, Y_orig_train, X_test, Y_test, Y_orig_test = load_raw_data(TRAIN_NPZ, TEST_NPZ)

    all_results = []
    for norm_value in norm_values:
        for seed in seeds:
            result = run_experiment(norm_value, seed,
                                     X_train, Y_train, Y_orig_train,
                                     X_test, Y_test, Y_orig_test,
                                     epochs=epochs)
            all_results.append(result)

    # Persist the full sweep so it can be analyzed later without re-training
    all_results = save_results(all_results, 'runs/vector_loss/sweep_results.json')

    # Quick summary: mean/std test accuracy per norm value, across seeds
    print("\n======== Sweep summary (test accuracy, mean ± std over seeds) ========")
    for norm_value in norm_values:
        accs = [r['test_accuracy'] for r in all_results if r['norm'] == norm_value]
        print(f"norm={norm_value}: {np.mean(accs):.4f} ± {np.std(accs):.4f}  (n={len(accs)} seeds)")