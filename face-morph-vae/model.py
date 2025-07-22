# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Residual Block to enhance feature learning in the decoder
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


# VAE model for UTKFace dataset
class VAE_UTKFace(nn.Module):
    def __init__(self, input_channel, encoding_size, **kwargs):
        super(VAE_UTKFace, self).__init__(**kwargs)
        
        # Encoder - Convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3, stride=2, padding=1),  # (h, w) = (32, 32)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # (16, 16)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # (8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # (4, 4)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256*4*4, 2*encoding_size)  # output: mean and log variance
        )
        
        # Decoder 1 (non-residual)
        self.decoder1 = nn.Sequential(
            nn.Linear(encoding_size, 128*4*4),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1, padding=1),  # (4, 4)
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (8, 8)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # (32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)  # (64, 64)
        )
        
        # Decoder 2 (residual)
        self.decoder2 = nn.Sequential(
            nn.Linear(encoding_size, 128*4*4),
            nn.Unflatten(1, (128, 4, 4)),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=1, padding=1),
            ResidualBlock(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResidualBlock(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResidualBlock(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResidualBlock(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def encoder_output(self, x):
        # Returns the encoder's output (mean and log variance)
        return self.encoder(x).chunk(2, dim=1)

    def reparametrize(self, dist_mean, logvar):
        # Reparameterization trick: Sampling from the distribution
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return dist_mean + eps * std

    def forward(self, x, skip_connect=None):
        # VAE forward pass
        mu, logvar = self.encoder_output(x)
        z = self.reparametrize(mu, logvar)
        x_recon = self.decoder1(z)  # Reconstructed image from latent space
        return x_recon, mu, logvar
