import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import PIL.Image

from model import VAE_UTKFace, ResidualBlock
from utils import (
    train_model,
    show_reconstructions,
    sample_from_latent,
    show_interpolation_candidates,
    interpolate_faces,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Seed for reproducibility
seed = 68
torch.manual_seed(seed)

# Hyperparameters and paths
BATCH_SIZE = 64
encoding_size = 100
input_channel = 3
learning_rate = 1e-3
num_epochs = 50
image_path = "/kaggle/input/utkface-new/UTKFace"

# Data transform pipeline
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Custom Dataset class (no labels)
class UTKFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = [f for f in os.listdir(root_dir) if f.endswith('.jpg') or f.endswith('.png')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.images[idx])
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Prepare dataset and dataloaders
dataset = UTKFaceDataset(image_path, transform=transform)
val_size = int(0.1 * len(dataset))  # 10% validation split
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# Model, loss, optimizer, scheduler
model = VAE_UTKFace(input_channel, encoding_size).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Train the model
train_loss, val_loss = train_model(model, num_epochs, train_dl, val_dl, loss_fn, optimizer, clip_norm=True, max_norm=50.0)

# Plot training curves
plt.plot(train_loss, label='Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Visualize reconstructions and samples from latent space
show_reconstructions(model, val_dl)
sample_from_latent(model)

# Pick two random faces from validation set for interpolation demo
val_iter = iter(val_dl)
val_batch = next(val_iter)
img1, img2 = val_batch[0], val_batch[1]

# Show candidates for interpolation
show_interpolation_candidates(img1, img2)

# Perform and display latent interpolation
interp_grid = interpolate_faces(model, img1, img2, steps=10)
plt.figure(figsize=(14, 2))
plt.axis("off")
plt.title("Final Latent Interpolation After Training")
plt.imshow(interp_grid.permute(1, 2, 0).cpu())
plt.show()

# Show reconstructions from random noise inputs
sample_input = torch.randn(20, 3, 64, 64).to(device)
show_reconstructions(model, sample_input=sample_input)
