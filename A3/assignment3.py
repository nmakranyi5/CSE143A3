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
import argparse


# Model constants.
BATCH_SIZE = 32
MAX_FEATURES = 20 * 1000
EMBEDDING_DIM = 128
SEQUENCE_LENGTH = 500
HIDDEN_LAYER_DIM = 64
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.001
EPOCHS = 15


## Loading the "20newsgroups" dataset.
def load_textfiles():
    RANDOM_SEED = 1337

    batch_size = BATCH_SIZE
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


vectorize_layer = keras.layers.TextVectorization(
    standardize="lower_and_strip_punctuation",
    max_tokens=MAX_FEATURES,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def build_model(args):
    inputs = keras.Input(shape=(None,), dtype="int64")

    #########################
    x = layers.Embedding(MAX_FEATURES, EMBEDDING_DIM)(inputs)
    x = layers.Dropout(DROPOUT_RATE)(x)

    if args.model == 'regularRnn':
        x = layers.Bidirectional(layers.SimpleRNN(HIDDEN_LAYER_DIM, activation="tanh"))(x)
        # x = layers.SimpleRNN(HIDDEN_LAYER_DIM, activation="tanh")(x)
    elif args.model == 'lstm':
        x = layers.Bidirectional(layers.LSTM(HIDDEN_LAYER_DIM, activation="tanh"))(x)
        # x = layers.LSTM(HIDDEN_LAYER_DIM, activation="tanh")(x)
    else:
        raise NotImplementedError()
    
    x = layers.Dense(HIDDEN_LAYER_DIM, activation="tanh")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    predictions = layers.Dense(20, activation="softmax", name="predictions")(x)
    #########################
    
    model = keras.Model(inputs, predictions)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"]
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='regularRnn',
                        choices=['regularRnn', 'lstm'])
    parser.add_argument('--do-test-set', dest='do_test_set', action='store_true')
    parser.add_argument('--no-do-test-set', dest='do_test_set', action='store_false')
    args = parser.parse_args()

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

    model = build_model(args)

    epochs = EPOCHS
    # Actually perform training.
    model.fit(train_ds, validation_data=val_ds, epochs=epochs)

    print("Evaluating on val set...")
    model.evaluate(val_ds)

    if args.do_test_set:
        print("Evaluating on test set...")
        model.evaluate(test_ds)
    
    print("Evaluating on train set...")
    model.evaluate(train_ds)


if __name__ == "__main__":
    main()
