#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Parse arguments
# TODO: Set reasonable defaults and possibly add more arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=64, type=int, help="Batch size.")
parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers.")
parser.add_argument("--units", default=100, type=int, help="Number of units in the hidden layers.")
parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")
args = parser.parse_args()

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

# Load the data
observations, labels = [], []
with open("gym_cartpole-data.txt", "r") as data:
    for line in data:
        columns = line.rstrip("\n").split()
        observations.append([float(column) for column in columns[0:-1]])
        labels.append(int(columns[-1]))
observations, labels = np.array(observations), np.array(labels)

# TODO: Create the model in the `model` variable.
# However, beware that there is currently a bug in Keras which does
# not correctly serialize InputLayer. Instead of using an InputLayer,
# pass explicitly `input_shape` to the first real model layer.
model = tf.keras.Sequential([
    layers.Dense(args.units, activation="relu", input_shape=[4,]),

    *[layers.Dense(args.units, activation="relu")
        for i in range(args.layers)],

    layers.Dense(4, activation="softmax"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

tb_callback=tf.keras.callbacks.TensorBoard(args.logdir)
model.fit(observations, labels, batch_size=args.batch_size, epochs=args.epochs, callbacks=[tb_callback])

model.save("gym_cartpole_model.h5", include_optimizer=False)
