# demo_script.py

import numpy as np
import tensorflow as tf
from tensorflow import keras

from model import AutoEncoder, VariationalAutoencoder
from utils import (
    compute_loss,
    train_model,
    ae_model_reconstruct_images,
    generate_images,
)

# Reproducibility
seed = 1
keras.backend.clear_session()
tf.random.set_seed(seed)
np.random.seed(seed)

# Load & preprocess Fashion MNIST
fashion_mnist = tf.keras.datasets.fashion_mnist.load_data()
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
X_train_full = X_train_full.astype(np.float32) / 255.0
X_test = X_test.astype(np.float32) / 255.0

X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

# Data pipeline
batch_size = 64
train_ds = tf.data.Dataset.from_tensor_slices(X_train).shuffle(50000).batch(batch_size, drop_remainder=True)
valid_ds = tf.data.Dataset.from_tensor_slices(X_valid).shuffle(10000).batch(batch_size, drop_remainder=True)

# Model setup
num_epochs = 20
coding_size = 10
input_shape = X_train.shape[1:]

# === VAE ===
vae_model = VariationalAutoencoder(coding_size, input_shape)
vae_optimizer = keras.optimizers.Adam(learning_rate=1e-3)
sample_latent_vector = tf.random.normal(shape=(16, coding_size))

# Before training
generate_images(vae_model, sample_latent_vector, model_status='before')

# Train VAE
train_model(vae_model, vae_optimizer, train_ds, valid_ds, num_epochs, sample_latent_vector)

# After training
generate_images(vae_model, sample_latent_vector, model_status='after')

# Latent interpolation (VAE)
print("\nInterpolating over the latent space...")
codings_grid = tf.reshape(sample_latent_vector, (1, 4, 4, coding_size))
larger_grid = tf.image.resize(codings_grid, size=[5, 7])
interpolated_codings = tf.reshape(larger_grid, [-1, coding_size])
images = vae_model.decoder(interpolated_codings).numpy()

import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
for i in range(len(images)):
    plt.subplot(5, 7, i + 1)
    plt.imshow(images[i], cmap='binary')
    plt.axis('off')
plt.tight_layout()
plt.show()

# Latent dimension exploration
print("\nExploring latent dimensions...")
for dim in range(coding_size):
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i, val in enumerate(np.linspace(-3, 3, 10)):
        z = np.zeros((1, coding_size))
        z[0, dim] = val
        img = vae_model.decoder(z)[0].numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].axis('off')
    plt.suptitle(f'Latent dimension {dim}')
    plt.show()

# === AE ===
ae_model = AutoEncoder(input_shape)
ae_model.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy']
)

# AE before training
ae_model_reconstruct_images(ae_model, X_train, model_status='before')

# Train AE
ae_model.fit(X_train, X_train, epochs=num_epochs, shuffle=True, validation_data=(X_valid, X_valid), verbose=1)

# AE after training
ae_model_reconstruct_images(ae_model, X_train, model_status='after')
