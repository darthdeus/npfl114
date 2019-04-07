#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf

from uppercase_data import UppercaseData

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--alphabet_size", default=30, type=int, help="If nonzero, limit alphabet to this many most frequent chars.")
parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")

parser.add_argument("--units", default=128, type=int, help="Number of hidden units.")
parser.add_argument("--layers", default=2, type=int, help="Number of hidden layers.")
parser.add_argument("--embedding", default=32, type=int, help="Embedding size.")
parser.add_argument("--dropout", default=0.0, type=float, help="Dropout rate.")

parser.add_argument("--hidden_layers", default="128,128", type=str, help="Hidden layer configuration.")
parser.add_argument("--threads", default=8, type=int, help="Maximum number of threads to use.")
parser.add_argument("--window", default=3, type=int, help="Window size to use.")

parser.add_argument("--dev_out_fname", default="tmp/uppercase_dev_out.txt", type=str, help="Name of the output file.")
parser.add_argument("--test_out_fname", default="tmp/uppercase_test_out.txt", type=str, help="Name of the output file.")

args = parser.parse_args()
args.hidden_layers = [int(hidden_layer) for hidden_layer in args.hidden_layers.split(",") if hidden_layer]

# Fix random seeds
np.random.seed(42)
tf.random.set_seed(42)
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Create logdir name
args.logdir = os.path.join("logs", "{}-{}-{}".format(
    os.path.basename(__file__),
    datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
    ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
))

# Load data
uppercase_data = UppercaseData(args.window, args.alphabet_size)

# TODO: Implement a suitable model, optionally including regularization, select
# good hyperparameters and train the model.
#
# The inputs are _windows_ of fixed size (`args.window` characters on left,
# the character in question, and `args.window` characters on right), where
# each character is representedy by a `tf.int32` index. To suitably represent
# the characters, you can:
# - Convert the character indices into _one-hot encoding_. There is no
#   explicit Keras layer, so you can
#   - use a Lambda layer which can encompass any function:
#       Sequential([
#         tf.layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
#         tf.layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet))),
#   - or use Functional API and a code looking like
#       inputs = tf.keras.layers.Input(shape=[2 * args.window + 1], dtype=tf.int32)
#       encoded = tf.one_hot(inputs, len(uppercase_data.train.alphabet))
#   You can then flatten the one-hot encoded windows and follow with a dense layer.
# - Alternatively, you can use `tf.keras.layers.Embedding`, which is an efficient
#   implementation of one-hot encoding followed by a Dense layer, and flatten afterwards.

from tensorflow.keras import layers

model = tf.keras.Sequential([
    layers.InputLayer(input_shape=[2 * args.window + 1], dtype=tf.int32),
    layers.Embedding(args.alphabet_size, args.embedding, input_length=args.window),
    layers.Flatten(),
    # layers.Lambda(lambda x: tf.one_hot(x, len(uppercase_data.train.alphabet), axis=1)),

])

for _ in range(args.layers):
    model.add(layers.Dense(args.units, activation=tf.nn.relu))
    model.add(layers.Dropout(args.dropout))

model.add(layers.Dense(2, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

model.fit(uppercase_data.train.data["windows"],
        uppercase_data.train.data["labels"],
        batch_size=args.batch_size,
        epochs=args.epochs)

def predict_data(dataset, fname):
    with open(fname, "w", encoding="utf-8") as out_file:
        preds = model.predict(dataset.data["windows"], batch_size=args.batch_size)
        preds = np.argmax(preds, axis=1)

        capitalized_text = "".join([c.capitalize() if u else c for c, u in zip(dataset.text.lower(), preds)])

        print(capitalized_text, file=out_file)

predict_data(uppercase_data.dev, args.dev_out_fname)
predict_data(uppercase_data.test, args.test_out_fname)
