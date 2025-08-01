import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_MNIST(nn.Module):
    def __init__(self, input_channel, encoding_size, drop_rate=0.2, multiple=4, skip_connect=False):
        """
        Variational Autoencoder for MNIST images with optional skip connections.
        """
        super().__init__()
        self.skip_connect = skip_connect

        # Encoder layers
        self.enc_1 = nn.Sequential(
            nn.Conv2d(input_channel, 32 * multiple, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32 * multiple), nn.ReLU(), nn.Dropout(p=drop_rate),
            nn.Conv2d(32 * multiple, 32 * multiple, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32 * multiple), nn.ReLU(), nn.Dropout(p=drop_rate),
        )  # Output: 14x14

        self.enc_2 = nn.Sequential(
            nn.Conv2d(32 * multiple, 64 * multiple, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64 * multiple), nn.ReLU(), nn.Dropout(p=drop_rate)
        )  # Output: 7x7

        # Latent space
        self.flatten = nn.Flatten()
        self.fc_mu_logvar = nn.Linear(64 * multiple * 7 * 7, 2 * encoding_size)

        # Decoder (vanilla)
        self.decoder_vanilla = nn.Sequential(
            nn.Linear(encoding_size, 64 * multiple * 7 * 7),
            nn.Unflatten(1, (64 * multiple, 7, 7)), nn.ReLU(),
            nn.ConvTranspose2d(64 * multiple, 64 * multiple, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64 * multiple), nn.ReLU(),
            nn.ConvTranspose2d(64 * multiple, 64 * multiple, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64 * multiple), nn.ReLU(),
            nn.ConvTranspose2d(64 * multiple, 32 * multiple, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32 * multiple, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

        if skip_connect:
            self.project_z = nn.Sequential(
                nn.Linear(encoding_size, 64 * multiple * 7 * 7),
                nn.Unflatten(1, (64 * multiple, 7, 7)), nn.ReLU()
            )

            self.up_1 = nn.Sequential(
                nn.ConvTranspose2d(64 * multiple, 64 * multiple, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            )

            self.dec_1 = nn.Sequential(
                nn.ConvTranspose2d(64 * multiple + 32 * multiple, 32 * multiple, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU()
            )

            self.dec_out = nn.Conv2d(32 * multiple, 1, kernel_size=3, padding=1)

    def encoder_output(self, x):
        x1 = self.enc_1(x)
        x2 = self.enc_2(x1)
        flat = self.flatten(x2)
        mu_logvar = self.fc_mu_logvar(flat)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        if self.skip_connect:
            return mu, logvar, (x1,)
        else:
            return mu, logvar, None

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        if self.skip_connect:
            mu, logvar, skips = self.encoder_output(x)
            z = self.reparametrize(mu, logvar)
            x = self.project_z(z)
            x = self.up_1(x)
            x = self.dec_1(torch.cat([x, skips[0]], dim=1))
            x = self.dec_out(x)
        else:
            mu, logvar, _ = self.encoder_output(x)
            z = self.reparametrize(mu, logvar)
            x = self.decoder_vanilla(z)
        return x, mu, logvar

    def forward_from_latent(self, z):
        """
        Decode from a latent vector z directly.
        """
        if self.skip_connect:
            x = self.project_z(z)
            x = self.up_1(x)
            B = z.size(0)
            skip_32 = torch.zeros((B, 32, 14, 14), device=z.device)
            x = self.dec_1(torch.cat([x, skip_32], dim=1))
            x = self.dec_out(x)
        else:
            x = self.decoder_vanilla(z)
        return x, None, None
