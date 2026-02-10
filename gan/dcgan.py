from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras import layers


@dataclass(frozen=True)
class DCGANConfig:
    img_size: int = 32
    channels: int = 3
    latent_dim: int = 128
    g_lr: float = 2e-4
    d_lr: float = 2e-4
    beta_1: float = 0.5


def build_generator(cfg: DCGANConfig) -> tf.keras.Model:
    # Output: [0,1] via sigmoid for consistency with normalized images
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(cfg.latent_dim,)),
            layers.Dense(4 * 4 * 256, use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Reshape((4, 4, 256)),
            layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2DTranspose(32, 4, strides=2, padding="same", use_bias=False),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(cfg.channels, 3, strides=1, padding="same", activation="sigmoid"),
        ],
        name="generator",
    )
    return model


def build_discriminator(cfg: DCGANConfig) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(cfg.img_size, cfg.img_size, cfg.channels)),
            layers.Conv2D(64, 4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, 4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Conv2D(256, 4, strides=2, padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1),
        ],
        name="discriminator",
    )
    return model


class DCGAN(tf.keras.Model):
    def __init__(self, generator: tf.keras.Model, discriminator: tf.keras.Model, cfg: DCGANConfig):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.cfg = cfg

        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.g_lr, beta_1=cfg.beta_1)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.d_lr, beta_1=cfg.beta_1)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_tracker, self.g_loss_tracker]

    def compile(self, *args, **kwargs):
        # keep custom optimizers/loss
        super().compile(*args, **kwargs)

    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        latent = tf.random.normal((batch_size, self.cfg.latent_dim))

        with tf.GradientTape() as d_tape:
            fake_images = self.generator(latent, training=True)
            real_logits = self.discriminator(real_images, training=True)
            fake_logits = self.discriminator(fake_images, training=True)

            real_labels = tf.ones_like(real_logits)
            fake_labels = tf.zeros_like(fake_logits)

            d_loss_real = self.loss_fn(real_labels, real_logits)
            d_loss_fake = self.loss_fn(fake_labels, fake_logits)
            d_loss = d_loss_real + d_loss_fake

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        latent = tf.random.normal((batch_size, self.cfg.latent_dim))
        with tf.GradientTape() as g_tape:
            fake_images = self.generator(latent, training=True)
            fake_logits = self.discriminator(fake_images, training=True)
            # generator wants discriminator to predict real
            g_loss = self.loss_fn(tf.ones_like(fake_logits), fake_logits)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        self.d_loss_tracker.update_state(d_loss)
        self.g_loss_tracker.update_state(g_loss)
        return {"d_loss": self.d_loss_tracker.result(), "g_loss": self.g_loss_tracker.result()}
