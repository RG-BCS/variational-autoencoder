# utils.py

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from scipy.stats import mode
import seaborn as sns

# --- VAE Losses ---

def compute_vae_loss_(x, x_recon, z_mean, z_logvar, image_size=784):
    recon_loss = keras.losses.BinaryCrossentropy()(x, x_recon) * image_size
    kl_loss = -0.5 * tf.reduce_sum(1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar), axis=-1)
    return tf.reduce_mean(recon_loss + kl_loss), recon_loss, kl_loss

def compute_total_loss(input_img, recon_output, mu, logvar,
                       beta=1.0, free_bits=0.0, loss_type='bce', verbose=False):
    if loss_type == 'bce':
        recon_loss_fn = keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
        recon_loss = tf.reduce_mean(recon_loss_fn(input_img, recon_output))
    elif loss_type == 'mse':
        recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(input_img - recon_output), axis=1))
    else:
        raise ValueError("loss_type must be 'bce' or 'mse'")

    kl = -0.5 * (1 + logvar - tf.square(mu) - tf.exp(logvar))
    if free_bits > 0:
        kl = tf.maximum(kl, free_bits)
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
    total_loss = recon_loss + beta * kl_loss

    if verbose:
        print(f"[Loss] total: {total_loss.numpy():.4f} | recon: {recon_loss.numpy():.4f} | "
              f"KL: {kl_loss.numpy():.4f} | β: {beta:.2f}")
    return total_loss, recon_loss, beta * kl_loss

# --- Training Helper ---

def grad_norm(gradients):
    norm = 0.0
    for g in gradients:
        if g is not None:
            norm += tf.reduce_sum(tf.square(g))
    return tf.sqrt(norm)

def train_vae(model, optimizer, dataset, num_epochs, compute_loss_fn, x_test=None, y_test=None,loss_type='bce', 
              beta_schedule="cyclic", cycle_length=20,warmup_frac=0.3, max_beta=1.0,free_bits=0.5):
  
    recon_losses, kl_losses, grad_norms = [], [], []
    def get_beta(epoch):
        if beta_schedule == "cyclic":
            return min(1.0, (epoch % cycle_length) / (cycle_length / 2))
        elif beta_schedule == "linear":
            ramp_up_epochs = int(num_epochs * warmup_frac)
            return max_beta * (epoch / ramp_up_epochs) if epoch < ramp_up_epochs else max_beta
        else:
            return max_beta  # constant β
    start = time.time()
    for epoch in range(num_epochs):
        total_loss = 0.0
        epoch_recon, epoch_kl = [], []
        epoch_grad_norms = []
        beta = get_beta(epoch)
        #beta = 1
        for x_batch in dataset:
            with tf.GradientTape() as tape:
                x_recon, z_mean, z_logvar, _ = model(x_batch)
                #loss, recon_loss, kl_loss = compute_loss_fn(x_batch, x_recon, z_mean, z_logvar, beta=beta,
                #                                            free_bits=free_bits,loss_type=loss_type)
                loss, recon_loss, kl_loss = compute_vae_loss_(x_batch,x_recon,z_mean,z_logvar)
                epoch_recon.append(tf.reduce_mean(recon_loss))
                epoch_kl.append(tf.reduce_mean(kl_loss))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            total_loss += loss
            
            batch_grad_norm = grad_norm(gradients)      # Gradient norm tracking
            epoch_grad_norms.append(batch_grad_norm)
            
        avg_loss = total_loss / len(dataset)
        avg_grad_norm = tf.reduce_mean(epoch_grad_norms)
        
        recon_losses.append(np.mean(epoch_recon))
        kl_losses.append(np.mean(epoch_kl))
        grad_norms.append(avg_grad_norm.numpy())
        elapsed = (time.time()-start)/60
        print(f'Epoch {epoch+1}/{num_epochs} | β={beta:.4f} | Loss {avg_loss:.4f} | recon {recon_losses[-1]:.4f} |'
              f' KL {kl_losses[-1]:.4f} | grad_norm {avg_grad_norm:.4f} | time_elapsed = {elapsed:.4f}min')
        start=time.time()
    return recon_losses, kl_losses ,grad_norms

# --- Visualization Utilities ---

def plot_latent_distribution(model, x_test, y_test, batch_size=100, title='Latent Space'):
    _, z_mean, _, _ = model.predict(x_test, batch_size=batch_size, verbose=0)
    plt.figure(figsize=(8, 8))
    markers = ('o', 'x', '^', '<', '>', '*', 'h', 'H', 'D', 'd', 'P', 'X', '8', 's', 'p')
    for i in np.unique(y_test):
        plt.scatter(z_mean[y_test == i, 0], z_mean[y_test == i, 1],
                    label=str(i), alpha=0.7, s=30,
                    marker=MarkerStyle(markers[i % len(markers)], fillstyle='none'))
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title(title)
    plt.legend(title='Digit Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def plot_generated_images(decoder, grid_dim=15, latent_dim=2, dim1=0, dim2=1):
    digit_size = 28
    figure = np.zeros((digit_size * grid_dim, digit_size * grid_dim))
    grid_x = np.linspace(-4, 4, grid_dim)
    grid_y = np.linspace(-4, 4, grid_dim)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.zeros((1, latent_dim))
            z_sample[0, dim1] = xi
            z_sample[0, dim2] = yi
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(12, 12))
    plt.imshow(figure, cmap='Greys_r')
    plt.xlabel(f"z[{dim1}]")
    plt.ylabel(f"z[{dim2}]")
    plt.title("Generated Digits from Latent Space")
    plt.grid(False)
    plt.show()

# --- K-Means Clustering & Evaluation ---

def relabel_clusters(y_pred, y_true):
    new_labels = np.zeros_like(y_pred)
    for cluster in range(10):
        mask = (y_pred == cluster)
        if np.any(mask):
            new_labels[mask] = mode(y_true[mask])[0]
    return new_labels

def plot_confusion_matrix(y_true, y_pred_relabel):
    conf_mat = confusion_matrix(y_true, y_pred_relabel)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix: True vs. K-means Clusters")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def plot_tsne(z_mean, y_labels, title="t-SNE of Latent Space"):
    tsne = TSNE(n_components=2, random_state=0, perplexity=30)
    z_tsne = tsne.fit_transform(z_mean)

    plt.figure(figsize=(10, 6))
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y_labels, cmap="tab10", s=5)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()
