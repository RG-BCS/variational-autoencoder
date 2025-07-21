# model.py

import tensorflow as tf
from tensorflow import keras

class AutoEncoder(keras.Model):
    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.encoder_ = keras.Sequential([
            keras.layers.InputLayer(shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(150, activation='selu', kernel_initializer='lecun_normal'),
            keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal')
        ])
        self.decoder_ = keras.Sequential([
            keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal'),
            keras.layers.Dense(150, activation='selu', kernel_initializer='lecun_normal'),
            keras.layers.Dense(784, activation='sigmoid'),
            keras.layers.Reshape((28, 28))
        ])

    def call(self, x):
        latent_space = self.encoder_(x)
        output_ = self.decoder_(latent_space)
        return output_


class VariationalAutoencoder(keras.Model):
    def __init__(self, coding_size, input_shape, **kwargs):
        super().__init__(**kwargs)
        self.encoder = keras.Sequential([
            keras.layers.InputLayer(shape=input_shape),
            keras.layers.Flatten(),
            keras.layers.Dense(150, activation='selu', kernel_initializer='lecun_normal'),
            keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal')
        ])
        self.coding_mean = keras.layers.Dense(coding_size)
        self.coding_log_var = keras.layers.Dense(coding_size)

        self.decoder = keras.Sequential([
            keras.layers.Dense(100, activation='selu', kernel_initializer='lecun_normal'),
            keras.layers.Dense(150, activation='selu', kernel_initializer='lecun_normal'),
            keras.layers.Dense(28 * 28, activation='sigmoid'),
            keras.layers.Reshape((28, 28))
        ])

    def call(self, x):
        x = self.encoder(x)
        coding_mean = self.coding_mean(x)
        coding_log_var = self.coding_log_var(x)
        latent_vector = self.reparameterize(coding_mean, coding_log_var)
        reconstruction = self.decoder(latent_vector)
        return reconstruction, coding_mean, coding_log_var

    def reparameterize(self, coding_mean, coding_log_var):
        epsilon = tf.random.normal(shape=coding_log_var.shape)
        return coding_mean + epsilon * tf.exp(coding_log_var / 2.0)
