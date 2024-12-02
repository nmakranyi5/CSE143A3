#!/usr/bin/env python3

"""
Assignment 3 starter code!

Based largely on:
    https://github.com/keras-team/keras-io/blob/master/examples/nlp/text_classification_from_scratch.py
"""

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
import numpy as np
from keras import layers


## Loading the "20newsgroups" dataset.
def load_textfiles():
    RANDOM_SEED = 1337

    batch_size = 32
    raw_train_ds = keras.utils.text_dataset_from_directory(
        "20_newsgroups",
        batch_size=batch_size,
        validation_split=0.2,
        subset="training",
        seed=RANDOM_SEED,
    )
    raw_val_ds = keras.utils.text_dataset_from_directory(
        "20_newsgroups",
        batch_size=batch_size,
        validation_split=0.2,
        subset="validation",
        seed=RANDOM_SEED,
    )

    raw_test_ds = keras.utils.text_dataset_from_directory(
        "20_newsgroups_test",
        batch_size=batch_size,
        seed=RANDOM_SEED,
    )

    print(f"Number of batches in raw_train_ds: {raw_train_ds.cardinality()}")
    print(f"Number of batches in raw_val_ds: {raw_val_ds.cardinality()}")
    print(f"Number of batches in raw_test_ds: {raw_test_ds.cardinality()}")
    return raw_train_ds, raw_val_ds, raw_test_ds


# Model constants.
max_features = 20 * 1000
embedding_dim = 128
sequence_length = 500

vectorize_layer = keras.layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def build_model():
    """
    ## Build a model

    We choose a simple 1D convnet starting with an `Embedding` layer.
    """
    # A integer input for vocab indices.
    inputs = keras.Input(shape=(None,), dtype="int64")

    # Next, we add a layer to map those vocab indices into a space of dimensionality
    # 'embedding_dim'.
    x = layers.Embedding(max_features, embedding_dim)(inputs)
    x = layers.Dropout(0.5)(x)

    # Conv1D + global max pooling
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
    x = layers.GlobalMaxPooling1D()(x)

    # We add a vanilla hidden layer:
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    # 20 possible output classes for the usenet dataset.
    predictions = layers.Dense(20, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs, predictions)

    # Compile the model with binary crossentropy loss and an adam optimizer.
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return model


def main():
    raw_train_ds, raw_val_ds, raw_test_ds = load_textfiles()

    # set the vocabulary!
    text_ds = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(text_ds)

    # Vectorize the data.
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    # Do async prefetching / buffering of the data for best performance on GPU.
    train_ds = train_ds.cache().prefetch(buffer_size=10)
    val_ds = val_ds.cache().prefetch(buffer_size=10)
    test_ds = test_ds.cache().prefetch(buffer_size=10)

    model = build_model()

    epochs = 10
    # Actually perform training.
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    """
    ## Evaluate the model on the test set or validation set.
    """
    ## model.evaluate(test_ds)
    model.evaluate(val_ds)


if __name__ == "__main__":
    main()
