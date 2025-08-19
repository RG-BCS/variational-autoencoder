import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

from model import NonConvVAE
from utils import (
    compute_total_loss,
    plot_latent_distribution,
    plot_generated_images,
    relabel_clusters,
    train_vae,
)

# --- Set seed for reproducibility ---
seed = 20
keras.backend.clear_session()
tf.random.set_seed(seed)
np.random.seed(seed)

# --- Hyperparameters ---
num_epochs = 30
latent_dim = 10
loss_type = 'bce'
beta_schedule = 'linear'
learning_rate = 1e-3
max_beta = 1.0
free_bits = 0.5
batch_size = 32

# --- Load and preprocess MNIST ---
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
image_size = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(-1, image_size).astype("float32") / 255.0
x_test = x_test.reshape(-1, image_size).astype("float32") / 255.0

x_train_dl = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000).batch(batch_size)
x_test_dl = tf.data.Dataset.from_tensor_slices(x_test).shuffle(1000).batch(batch_size)

# --- Initialize model and optimizer ---
vae_model = NonConvVAE(latent_dim=latent_dim)
optimizer = keras.optimizers.Adam(learning_rate)

# --- Visualize latent space before training ---
plot_latent_distribution(vae_model, x_test, y_test, batch_size=100, title='Before Training')

# --- Train the model ---
recon_losses, kl_losses, grad_norms = train_vae(
    vae_model,
    optimizer,
    x_train_dl,
    num_epochs,
    compute_loss_fn=compute_total_loss,
    beta_schedule=beta_schedule,
    max_beta=max_beta,
    free_bits=free_bits,
    loss_type=loss_type,
    verbose=True,
)

# --- Plot training curves ---
plt.figure()
plt.plot(recon_losses, label='Reconstruction Loss')
plt.plot(kl_losses, label='KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.title("Training Losses")
plt.show()

# --- Extract latent vectors ---
_, z_mean, _, _ = vae_model.predict(x_test, batch_size=100, verbose=0)

# --- Apply KMeans in latent space ---
kmeans = KMeans(n_clusters=10, random_state=0, n_init='auto')
y_pred = kmeans.fit_predict(z_mean)
y_pred_relabel = relabel_clusters(y_pred, y_test)

# --- Scatter plot of latent clusters ---
plt.figure(figsize=(10, 6))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_pred_relabel, cmap="tab10", s=5)
plt.title("K-means Clustering in Latent Space (Relabeled)")
plt.xlabel("z[0]")
plt.ylabel("z[1]")
plt.grid(True)
plt.show()

# --- Confusion matrix ---
conf_mat = confusion_matrix(y_test, y_pred_relabel)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: True vs. K-means Clusters (Relabeled)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# --- t-SNE projection ---
tsne = TSNE(n_components=2, random_state=0, perplexity=30)
z_tsne = tsne.fit_transform(z_mean)

plt.figure(figsize=(10, 6))
plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y_test, cmap="tab10", s=5)
plt.title("t-SNE Projection of Latent Space")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.grid(True)
plt.show()

# --- Final latent distribution ---
plot_latent_distribution(vae_model, x_test, y_test, batch_size=100, title='After Training')

# --- Sample from latent space ---
plot_generated_images(vae_model.decode, latent_dim=latent_dim)
