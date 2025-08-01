import torch
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Constants
BATCH_SIZE = 64
ANOMALY_CLASS = [5]  # '5' is considered abnormal
NORMAL_CLASS = list(range(10))
NORMAL_CLASSES = [c for c in NORMAL_CLASS if c not in ANOMALY_CLASS]

# Transform pipeline for MNIST
transform = transforms.Compose([transforms.ToTensor()])

# Load full MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def label_normal_abnormal(dataset, anomaly_digits):
    """
    Assign binary labels: 0 for normal, 1 for anomaly.
    """
    labels = np.array(dataset.targets)
    new_targets = np.isin(labels, anomaly_digits).astype(int)
    return new_targets

# Prepare training subset (normal digits only)
train_indices = [i for i, label in enumerate(train_dataset.targets) if label in NORMAL_CLASSES]
train_subset = Subset(train_dataset, train_indices)

# Create validation set from a portion of the training set
val_ratio = 0.1
val_size = int(len(train_subset) * val_ratio)
val_indices = np.random.choice(len(train_subset), val_size, replace=False)
val_subset = Subset(train_subset, val_indices)

# Relabel test set to 0 (normal) or 1 (anomalous)
test_targets = label_normal_abnormal(test_dataset, anomaly_digits=ANOMALY_CLASS)
test_dataset.targets = torch.tensor(test_targets)

# Dataloaders
train_dl = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
val_dl   = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_dl  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
