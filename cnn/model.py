from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from utils.config import IMG_SIZE, NUM_CLASSES


def build_cnn(num_classes: int = NUM_CLASSES, img_size: int = IMG_SIZE) -> tf.keras.Model:
    inputs = layers.Input(shape=(img_size, img_size, 3))

    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Conv2D(128, 3, padding="same")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gtsrb_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
