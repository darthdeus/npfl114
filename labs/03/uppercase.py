#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

ALPHABET_SIZE = 100


# Loads an uppercase dataset.
# - The dataset either uses a specified alphabet, or constructs an alphabet of
#   specified size consisting of most frequent characters.
# - The batches are generated using a sliding window of given size,
#   i.e., for a character, we generate left `window` characters, the character
#   itself and right `window` characters, 2 * `window` +1 in total.
# - The batches can be either generated using `next_batch`+`epoch_finished`,
#   or all data in the original order can be generated using `all_data`.
class Dataset:
    def __init__(self, filename, window, alphabet):
        self._window = window

        # Load the data
        with open(filename, "r", encoding="utf-8") as file:
            self._text = file.read()

        # Create alphabet_map
        alphabet_map = {"<pad>": 0, "<unk>": 1}
        if not isinstance(alphabet, int):
            for index, letter in enumerate(alphabet):
                alphabet_map[letter] = index
        else:
            # Find most frequent characters
            freqs = {}
            for char in self._text:
                char = char.lower()
                freqs[char] = freqs.get(char, 0) + 1

            most_frequent = sorted(freqs.items(), key=lambda item: item[1], reverse=True)
            for i, (char, freq) in enumerate(most_frequent, len(alphabet_map)):
                alphabet_map[char] = i
                if len(alphabet_map) >= alphabet: break

        # Remap input characters using the alphabet_map
        self._lcletters = np.zeros(len(self._text) + 2 * window, np.uint8)
        self._labels = np.zeros(len(self._text), np.bool)
        for i in range(len(self._text)):
            char = self._text[i].lower()
            if char not in alphabet_map: char = "<unk>"
            self._lcletters[i + window] = alphabet_map[char]
            self._labels[i] = self._text[i].isupper()

        # Compute alphabet
        self._alphabet = [""] * len(alphabet_map)
        for key, value in alphabet_map.items():
            self._alphabet[value] = key

        self._permutation = np.random.permutation(len(self._text))

    def _create_batch(self, permutation):
        batch_windows = np.zeros([len(permutation), 2 * self._window + 1], np.int32)
        for i in range(0, 2 * self._window + 1):
            batch_windows[:, i] = self._lcletters[permutation + i]

        # batch_windows = OneHotEncoder(n_values=ALPHABET_SIZE, sparse=False).fit_transform(batch_windows)
        return batch_windows, self._labels[permutation]

    @property
    def alphabet(self):
        return self._alphabet

    @property
    def text(self):
        return self._text

    @property
    def labels(self):
        return self._labels

    def all_data(self):
        return self._create_batch(np.arange(len(self._text)))

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._create_batch(batch_perm)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._text))
            return True
        return False


class Network:
    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            WINDOW_SIZE = (2 * args.window + 1)

            self.windows = tf.placeholder(tf.int32, [None, WINDOW_SIZE], name="windows")
            self.labels = tf.placeholder(tf.int32, [None], name="labels")  # Or you can use tf.int32
            self.is_training = tf.placeholder_with_default(False, (), name="is_training")

            with tf.name_scope("preprocessing"):
                hot = tf.one_hot(self.windows, args.alphabet_size, axis=1)
                # encoded = tf.reshape(hot, (-1, args.alphabet_size * WINDOW_SIZE))
                encoded = tf.layers.flatten(hot)
                hidden = tf.cast(encoded, dtype=tf.float32, name="hidden0")

            UNITS = args.units

            hist_hidden = []
            hist_relu = []

            # TODO: rozbalit, gradient super highway :D

            # hidden = tf.layers.dense(hidden, UNITS, activation=tf.nn.relu, name="hidden_pre")
            # for i in range(1):
            #     with tf.name_scope(f"hidden-{i}"):
            #         hidden = tf.layers.dense(hidden, UNITS, activation=tf.nn.relu, name=f"hidden-{i+1}")
            #         hidden = tf.layers.dropout(hidden, rate=0.3, training=self.is_training)
                    # hidden = tf.layers.batch_normalization(hidden, axis=1, training=self.is_training)

                # hist_hidden.append(hidden)
                # hist_relu.append(hidden)

            # hidden = tf.layers.dense(hidden, UNITS, activation=tf.nn.relu, name="hidden_post")

            output = tf.layers.dense(hidden, 2, activation=None, name="output")

            self.predictions = tf.argmax(output, axis=1, output_type=tf.int32, name="predictions")

            with tf.name_scope("loss"):
                loss = tf.losses.sparse_softmax_cross_entropy(logits=output, labels=self.labels)

            global_step = tf.train.create_global_step()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                # lr_decay = tf.train.exponential_decay(args.learning_rate, global_step, 1, args.lr_decay_rate, name="lr_decay")
                self.training = tf.train.AdamOptimizer(0.0005).minimize(loss, global_step=global_step)

            with tf.name_scope("accuracy"):
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))

            # Summaries
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
                summs = [
                    tf.contrib.summary.scalar("train/loss", loss),
                    tf.contrib.summary.scalar("train/accuracy", self.accuracy),
                ]

                # for i in range(len(hist_hidden)):
                #     summs.append(tf.contrib.summary.histogram(f"train/hidden-norelu-{i}", hist_hidden[i]))
                #     summs.append(tf.contrib.summary.histogram(f"train/hidden-relu-{i}", hist_relu[i]))

                self.summaries["train"] = summs
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in ["dev", "test"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
                                               tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, windows, labels):
        self.session.run([self.training, self.summaries["train"]],
                         {self.windows: windows, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, windows, labels):
        return self.session.run([self.accuracy, self.summaries[dataset]],
                                {self.windows: windows, self.labels: labels, self.is_training: False})


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--alphabet_size", default=ALPHABET_SIZE, type=int, help="Alphabet size.")
    parser.add_argument("--batch_size", default=4096, type=int, help="Batch size.")
    parser.add_argument("--learning_rate", default=0.0005, type=float, help="Learning rate.")
    parser.add_argument("--units", default=1024, type=int, help="Number of neurons in the hidden units.")
    parser.add_argument("--epochs", default=40, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=12, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--window", default=7, type=int, help="Size of the window to use.")
    args = parser.parse_args()

    assert args.alphabet_size == 100

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists("logs"): os.mkdir("logs")  # TF 1.6 will do this by itself

    # Load the data
    train = Dataset("data/uppercase_data_train.txt", args.window, alphabet=args.alphabet_size)
    dev = Dataset("data/uppercase_data_dev.txt", args.window, alphabet=train.alphabet)
    test = Dataset("data/uppercase_data_test.txt", args.window, alphabet=train.alphabet)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            windows, labels = train.next_batch(args.batch_size)
            network.train(windows, labels)

        dev_windows, dev_labels = dev.all_data()
        acc, _ = network.evaluate("dev", dev_windows, dev_labels)
        print(f"Dev acc: {acc}")

    # TODO: Generate the uppercased test set
