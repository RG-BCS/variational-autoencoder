import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from model import VAE_MNIST
from dataloader_generator import train_dl, val_dl, test_dl
from utils import (
    train_model, get_recon_losses_per_image_after_training, 
    show_reconstructions, plot_confusion_matrix, plot_roc_pr
)

# Set seed for reproducibility
seed = 15
torch.manual_seed(seed)

# Hyperparameters and config
encoding_size = 10
input_channel = 1
learning_rate = 1e-3
LOSS_TYPE = 'bce'
num_epochs = 300

index_to_name = {0: 'normal', 1: 'abnormal'}
name_to_index = {'normal': 0, 'abnormal': 1}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, loss, optimizer, scheduler
model = VAE_MNIST(input_channel, encoding_size, drop_rate=0.1, multiple=4, skip_connect=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Optional: Cosine Annealing LR scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Train the model
train_loss, val_loss, train_recon_loss, train_kl_loss = train_model(
    model, num_epochs, train_dl, val_dl, test_dl, optimizer,
    loss_type=LOSS_TYPE, scheduler=scheduler, clip_norm=True, max_norm=100.0
)

# Determine threshold from training set
train_losses = get_recon_losses_per_image_after_training(model, train_dl, LOSS_TYPE)
test_losses = get_recon_losses_per_image_after_training(model, test_dl, LOSS_TYPE)
THRESHOLD = train_losses.mean() + 2 * train_losses.std()

# Plot training loss curves
plt.plot(train_loss, label='Total Train Loss')
plt.plot(val_loss, label='Total Val Loss')
plt.plot(train_recon_loss, label='Recon Loss')
plt.plot(train_kl_loss, label='KL Loss')
plt.title("VAE Loss Components Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Visualize reconstructions for abnormal samples
sample_fives = []
for x, y in test_dl:
    r = x[y == 1]
    sample_fives.extend(r)
sample_fives_tensor = torch.stack(sample_fives)
show_reconstructions(model, dataloader=None, sample_input=sample_fives_tensor, num_images=20)

# Confusion Matrix
plot_confusion_matrix(
    model, test_dl, threshold=THRESHOLD,
    labels=['normal', 'abnormal'], normalize=False,
    title='Confusion Matrix', loss_type=LOSS_TYPE
)

# ROC and PR curves
plot_roc_pr(model, test_dl)

# Histogram of reconstruction losses
sns.histplot(train_losses, label='Train (normal)', stat='density', kde=True)
sns.histplot(test_losses, label='Test', stat='density', kde=True)
plt.axvline(THRESHOLD, color='red', linestyle='--', label='Threshold')
plt.title("Reconstruction Loss Distribution")
plt.legend()
plt.grid(True)
plt.show()
