# model.py

import tensorflow as tf
from tensorflow import keras

class NonConv_VAE(keras.Model):
    def __init__(self, intermediate_dim=512, latent_dim=2, image_size=784, **kwargs):
        super().__init__(**kwargs)
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        self.image_size = image_size

        # Encoder network
        self.encoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(image_size,)),
            keras.layers.Dense(intermediate_dim * 2, activation='relu'),
            keras.layers.Dense(intermediate_dim, activation='relu')
        ])

        self.z_mean = keras.layers.Dense(latent_dim, name="z_mean")
        self.z_logvar = keras.layers.Dense(latent_dim, name="z_logvar")

        # Decoder network
        self.decoder = keras.Sequential([
            keras.layers.InputLayer(input_shape=(latent_dim,)),
            keras.layers.Dense(intermediate_dim * 2, activation='relu'),
            keras.layers.Dense(intermediate_dim, activation='relu'),
            keras.layers.Dense(image_size, activation='sigmoid')
        ])

    def reparameterize(self, z_mean, z_logvar):
        eps = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_logvar) * eps

    def call(self, x):
        encoded = self.encoder(x)
        z_mean = self.z_mean(encoded)
        z_logvar = self.z_logvar(encoded)
        z = self.reparameterize(z_mean, z_logvar)
        x_recon = self.decoder(z)
        return x_recon, z_mean, z_logvar, z
