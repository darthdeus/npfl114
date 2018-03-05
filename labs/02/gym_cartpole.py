#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

class Network:
    OBSERVATIONS = 4
    ACTIONS = 2

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                       intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            self.observations = tf.placeholder(tf.float32, [None, self.OBSERVATIONS], name="observations")
            self.labels = tf.placeholder(tf.int64, [None], name="labels")

            hidden = self.observations

            self.training_flag = tf.placeholder_with_default(False, (), name="training_flag")

            for i in range(args.layers):
                with tf.name_scope("layer{}".format(i)):
                    hidden = tf.layers.dense(hidden, args.neurons, activation=tf.nn.relu)
                    hidden = tf.layers.dropout(hidden, training=self.training_flag)

            output_layer = tf.layers.dense(hidden, 2, activation=None)
            # TODO: Define the model, with the output layers for actions in `output_layer`

            self.actions = tf.argmax(output_layer, axis=1, name="actions")

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

            # Global step
            xentropy = tf.losses.sparse_softmax_cross_entropy(self.labels, output_layer, scope="xentropy")
            loss = tf.add_n([xentropy] + reg_losses, name="loss")

            global_step = tf.train.create_global_step()
            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.actions), tf.float32))

            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)

            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                self.summaries = {}

                for run in ["train", "test"]:
                    self.summaries[run] = [tf.contrib.summary.scalar("{}/loss".format(run), loss),
                                           tf.contrib.summary.scalar("{}/accuracy".format(run), self.accuracy)]

            # Construct the saver
            tf.add_to_collection("end_points/observations", self.observations)
            tf.add_to_collection("end_points/actions", self.actions)
            self.saver = tf.train.Saver()

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, observations, labels):
        acc, _, _ = self.session.run([self.accuracy, self.training, self.summaries["train"]], {self.observations: observations,
                                                                    self.labels: labels,
                                                                    self.training_flag: True})
        return acc

    def evaluate(self, observations, labels):
        acc, _ = self.session.run([self.accuracy, self.summaries["test"]], {self.observations: observations,
                                                                          self.labels: labels,
                                                                          self.training_flag: False})
        return acc

    def save(self, path):
        self.saver.save(self.session, path)


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=20, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--neurons", default=100, type=int, help="Number of neurons in the hidden layers.")
    parser.add_argument("--layers", default=1, type=int, help="Number of hidden layers.")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs") # TF 1.6 will do this by itself

    # Load the data
    observations, labels = [], []
    with open("gym_cartpole-data.txt", "r") as data:
        for line in data:
            columns = line.rstrip("\n").split()
            observations.append([float(column) for column in columns[0:4]])
            labels.append(int(columns[4]))
    observations, labels = np.array(observations), np.array(labels)

    idx = np.random.permutation(len(observations))

    split = 0.20

    train_size = int(1 - len(idx) * 0.2)
    train_idx = idx[:train_size]
    test_idx = idx[train_size:]

    # print("Splitting train: {}, test: {}".format(len(train_idx), len(test_idx)))

    assert len(train_idx) > len(test_idx)

    X_train, y_train = observations[train_idx], labels[train_idx]
    X_valid, y_valid = observations[test_idx], labels[test_idx]

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        train_acc = network.train(X_train, y_train)
        valid_acc = network.evaluate(X_valid, y_valid)
        if i % 20 == 0:
            print("Acc: {:.2f}\tValid: {:.2f}".format(train_acc, valid_acc))

    # Save the network
    network.save("gym_cartpole/model")
