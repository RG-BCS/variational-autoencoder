# utils.py
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

# Compute the total loss combining reconstruction and KL divergence
def compute_total_loss(input_img, recon_img, mu, logvar, loss_fn, beta=1.0):
    # Reconstruction error (BCE loss)
    recon_err = loss_fn(recon_img, input_img)
    # Regularization (KL divergence)
    regul_err = -0.5 * torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar), dim=-1)
    return torch.mean(recon_err + beta * regul_err)


# Another variation of total loss calculation for binary cross entropy
def compute_total_loss_v2(input_img, recon_logits, mu, logvar, beta=1.0, free_bits=0.0, verbose=False):
    # Binary cross entropy with raw logits (no sigmoid on decoder)
    recon_loss = F.binary_cross_entropy_with_logits(recon_logits, input_img, reduction='none')
    recon_loss = recon_loss.sum(dim=[1, 2, 3])  # per sample
    recon_loss = recon_loss.mean()

    # KL divergence between posterior and N(0,1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl = torch.clamp(kl, min=free_bits)
    kl_loss = kl.sum(dim=1).mean()  # sum over latent dim, average over batch

    total_loss = recon_loss + beta * kl_loss

    if verbose:
        print(f"[Loss] total: {total_loss.item():.4f} | recon: {recon_loss.item():.4f}"\
              f" | KL: {kl_loss.item():.4f} | β: {beta:.2f}")
    return total_loss


# Show the two faces selected for interpolation
def show_interpolation_candidates(img1, img2):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))  # slightly larger for breathing room
    axs[0].imshow(img1.permute(1, 2, 0).cpu())
    axs[0].set_title("Image 1")
    axs[0].axis("off")
    axs[1].imshow(img2.permute(1, 2, 0).cpu())
    axs[1].set_title("Image 2")
    axs[1].axis("off")
    plt.suptitle("Latent Interpolation Candidates", y=1.05)
    plt.tight_layout()
    plt.show()


# Interpolate between two latent points and show the results
def interpolate_faces(model, img1, img2, steps=10):
    was_training = model.training  # cache current mode
    model.eval()
    with torch.no_grad():
        z1_mu, _ = model.encoder_output(img1.unsqueeze(0).to(device))
        z2_mu, _ = model.encoder_output(img2.unsqueeze(0).to(device))

        # Linearly interpolate in latent space
        interpolations = []
        for alpha in torch.linspace(0, 1, steps):
            z = (1 - alpha) * z1_mu + alpha * z2_mu
            recon = model.decoder1(z.to(device))  # Reconstruction from interpolated latent vector
            interpolations.append(recon.squeeze(0))
        
        # Create a grid image
        grid = torchvision.utils.make_grid(interpolations, nrow=steps, normalize=True)
        if was_training:
            model.train()
        return grid


# Sample random images from latent space and show them
def sample_from_latent(model, num_samples=8):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, model.decoder1[0].in_features).to(device)
        samples = model.decoder1(z)  # Generate samples from random latent variables
        grid = torchvision.utils.make_grid(samples, nrow=num_samples, normalize=True)
        
        plt.figure(figsize=(20, 4))
        plt.imshow(grid.permute(1, 2, 0).cpu())
        plt.axis("off")
        plt.title("Samples from Latent Space")
        plt.show()


# Show original images and their reconstructions
def show_reconstructions(model, dataloader=None, sample_input=None, num_images=8):
    model.eval()
    with torch.no_grad():
        if dataloader:
            batch = next(iter(dataloader))
        else: 
            batch = sample_input
        
        batch = batch[:num_images].to(device)
        recon, _, _ = model(batch)
        
        recon = torch.sigmoid(recon)  # Ensure values in [0, 1]
        
        comparison = torch.cat([batch, recon])  # Concatenate original and reconstructed images
        grid = torchvision.utils.make_grid(comparison.cpu(), nrow=num_images, normalize=False, pad_value=1)
        plt.figure(figsize=(20, 4))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        if dataloader:
            plt.title("Top: Original | Bottom: Reconstructed")
        else:
            plt.title("Top: Input Noise | Bottom: Reconstructed")
        plt.show()


# Calculate gradient norm during training to track model's training progress
def grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


# Training function to train the VAE model
def train_model(model, num_epochs, train_dl, val_dl, loss_fn, optimizer,
                clip_norm=False, max_norm=1.0, sample_images_for_interp=None):
    
    def cyclical_beta(epoch, cycle_length=20):
        return min(1.0, (epoch % cycle_length) / (cycle_length / 2))

    model.train()
    total_train_loss = []
    total_val_loss = []
    
    # Pick two sample faces to interpolate (optional)
    if sample_images_for_interp is None:
        sample_iter = iter(train_dl)
        sample_batch = next(sample_iter)
        img1, img2 = sample_batch[0], sample_batch[1]
        show_interpolation_candidates(img1, img2)

    start_time = time.time()
    for epoch in range(num_epochs):
        beta = cyclical_beta(epoch)
        epoch_loss = 0.
        for input_images in train_dl:
            input_images = input_images.to(device)
            recon_img, mu, logvar = model(input_images)
            loss = compute_total_loss(input_images, recon_img, mu, logvar, loss_fn, beta)
            loss.backward()
            if clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            norm_grad = grad_norm(model)
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dl.dataset)
        total_train_loss.append(epoch_loss)
        
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for val_batch in val_dl:
                val_batch = val_batch.to(device)
                recon, mu, logvar = model(val_batch)
                loss = compute_total_loss(val_batch, recon, mu, logvar, loss_fn, beta)
                val_loss += loss.item()

        val_loss /= len(val_dl.dataset)
        total_val_loss.append(val_loss)
        elapsed_time = (time.time()-start_time) / 60
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Grad Norm: {norm_grad:.4f} | Time: {elapsed_time:.3f} min")
            
            start_time = time.time()
            # Visualize latent interpolation
            interp_grid = interpolate_faces(model, img1, img2, steps=10)
            plt.figure(figsize=(12, 2))
            plt.axis('off')
            plt.title(f'Latent Interpolation — Epoch {epoch + 1}')
            plt.imshow(interp_grid.permute(1, 2, 0).cpu())
            plt.show()
        
        model.train()

    return total_train_loss, total_val_loss
