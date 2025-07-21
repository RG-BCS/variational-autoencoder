# utils.py

import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

def compute_loss(input_image, recon_image, coding_mean, coding_log_var):
    recon_loss = keras.losses.binary_crossentropy(input_image, recon_image) * 784
    latent_loss = -0.5 * tf.reduce_sum(
        1 + coding_log_var - tf.square(coding_mean) - tf.exp(coding_log_var),
        axis=-1
    )
    return tf.reduce_mean(recon_loss + latent_loss)

def train_steps(real_image, model, optimizer):
    with tf.GradientTape() as tape:
        recon_image, coding_mean, coding_log_var = model(real_image)
        losses = compute_loss(real_image, recon_image, coding_mean, coding_log_var)
    gradients = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return losses

def train_model(model, optimizer, train_ds, valid_ds, num_epochs, sample_latent_vector):
    for epoch in range(num_epochs):
        train_loss = 0.
        valid_loss = 0.
        start = time.time()

        for real_image in train_ds:
            train_loss += train_steps(real_image, model, optimizer)
        train_loss /= len(train_ds)

        for real_image in valid_ds:
            valid_loss += compute_loss(real_image, *model(real_image))
        valid_loss /= len(valid_ds)

        print(f'Epoch {epoch+1}/{num_epochs} - {time.time()-start:.2f}s - Train Loss: {train_loss:.4f} - Val Loss: {valid_loss:.4f}')
        generate_images(model, sample_latent_vector, model_status=f'epoch_{epoch+1}')

def generate_images(model, sample_latent_vector, model_status='after'):
    plt.figure(figsize=(6, 4))
    plt.suptitle(f"{model_status.capitalize()} VAE Generated Images", fontsize=14, y=1.05)
    for i in range(len(sample_latent_vector)):
        recons = model.decoder(sample_latent_vector[i:i+1])[0, :, :]
        plt.subplot(4, 4, i + 1)
        plt.imshow(recons, cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def ae_model_reconstruct_images(model, X, model_status='before'):
    x = X[:10]
    y = model(X[:10])
    plt.figure(figsize=(14, 3))
    plt.suptitle(f"{model_status.capitalize()} AE Reconstruction", fontsize=14, y=1.05)

    for i in range(len(x)):
        plt.subplot(2, 10, i + 1)
        plt.imshow(x[i], cmap='binary')
        plt.title('Original', fontsize=8)
        plt.axis('off')

        plt.subplot(2, 10, i + 11)
        plt.imshow(y[i], cmap='binary')
        plt.title('Reconstructed', fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
