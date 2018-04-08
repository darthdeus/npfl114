# coding=utf-8

source_1 = """#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


class Dataset:
    def __init__(self, filename, shuffle_batches=True):
        data = np.load(filename)
        self._voxels = data[\"voxels\"]
        self._labels = data[\"labels\"] if \"labels\" in data else None

        self._shuffle_batches = shuffle_batches
        self._new_permutation()

    def _new_permutation(self):
        if self._shuffle_batches:
            self._permutation = np.random.permutation(len(self._voxels))
        else:
            self._permutation = np.arange(len(self._voxels))

    def split(self, ratio):
        split = int(len(self._voxels) * ratio)

        first, second = Dataset.__new__(Dataset), Dataset.__new__(Dataset)
        first._voxels, second._voxels = self._voxels[:split], self._voxels[split:]
        if self._labels is not None:
            first._labels, second._labels = self._labels[:split], self._labels[split:]
        else:
            first._labels, second._labels = None, None

        for dataset in [first, second]:
            dataset._shuffle_batches = self._shuffle_batches
            dataset._new_permutation()

        return first, second

    @property
    def voxels(self):
        return self._voxels

    @property
    def labels(self):
        return self._labels

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
        return self._voxels[batch_perm], self._labels[batch_perm] if self._labels is not None else None

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._new_permutation()
            return True
        return False


class Network:
    LABELS = 10

    def __init__(self, threads, seed=42):
        # Create an empty graph and a session
        graph = tf.Graph()
        graph.seed = seed
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
                                                                     intra_op_parallelism_threads=threads,
                                                                     gpu_options=gpu_options))

    def construct(self, args):
        with self.session.graph.as_default():
            # Inputs
            self.voxels = tf.placeholder(
                tf.float32, [None, args.modelnet_dim, args.modelnet_dim, args.modelnet_dim, 1], name=\"voxels\")
            self.labels = tf.placeholder(tf.int64, [None], name=\"labels\")
            self.is_training = tf.placeholder(tf.bool, [], name=\"is_training\")

            self.global_step = tf.train.create_global_step()

            # TODO: Computation and training.
            #
            # The code below assumes that:
            # - loss is stored in `loss`
            # - training is stored in `self.training`
            # - label predictions are stored in `self.predictions`
            features = self.voxels

            features = tf.layers.conv3d(features, 32, 3, 1)
            # features = tf.layers.batch_normalization(features, training=self.is_training)
            features = tf.nn.relu(features)
            features = tf.layers.max_pooling3d(features, 2, 1)

            features = tf.layers.conv3d(features, 32, 3, 1)
            # features = tf.layers.batch_normalization(features, training=self.is_training)
            features = tf.nn.relu(features)
            features = tf.layers.max_pooling3d(features, 2, 1)

            features = tf.layers.conv3d(features, 32, 3, 1)
            # features = tf.layers.batch_normalization(features, training=self.is_training)
            features = tf.nn.relu(features)
            features = tf.layers.max_pooling3d(features, 2, 1)

            features = tf.layers.flatten(features)

            features = tf.layers.dense(features, 1024)
            # features = tf.layers.dropout(features, training=self.is_training)
            features = tf.nn.relu(features)

            # features = tf.layers.dense(features, 1024)
            # features = tf.layers.dropout(features, training=self.is_training)
            # features = tf.nn.relu(features)

            scores = tf.layers.dense(features, 10)

            self.predictions = tf.argmax(scores, axis=1, name=\"predictions\")

            loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=scores)

            self.training = tf.train.AdamOptimizer().minimize(loss, global_step=self.global_step)

            # Summaries
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.labels, self.predictions), tf.float32))
            summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
            self.summaries = {}
            with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(8):
                self.summaries[\"train\"] = [tf.contrib.summary.scalar(\"train/loss\", loss),
                                           tf.contrib.summary.scalar(\"train/accuracy\", self.accuracy)]
            with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for dataset in [\"dev\", \"test\"]:
                    self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + \"/loss\", loss),
                                               tf.contrib.summary.scalar(dataset + \"/accuracy\", self.accuracy)]

            # Initialize variables
            self.session.run(tf.global_variables_initializer())
            with summary_writer.as_default():
                tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

    def train(self, voxels, labels):
        self.session.run([self.training, self.summaries[\"train\"]],
                         {self.voxels: voxels, self.labels: labels, self.is_training: True})

    def evaluate(self, dataset, voxels, labels):
        accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]],
                                       {self.voxels: voxels, self.labels: labels, self.is_training: False})
        return accuracy

    def predict(self, voxels):
        return self.session.run(self.predictions, {self.voxels: voxels, self.is_training: False})


if __name__ == \"__main__\":
    import argparse
    import datetime
    import os
    import re

    # Fix random seed
    np.random.seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(\"--batch_size\", default=32, type=int, help=\"Batch size.\")
    parser.add_argument(\"--epochs\", default=20, type=int, help=\"Number of epochs.\")
    parser.add_argument(\"--modelnet_dim\", default=20, type=int, help=\"Dimension of ModelNet data.\")
    parser.add_argument(\"--threads\", default=8, type=int, help=\"Maximum number of threads to use.\")
    parser.add_argument(\"--train_split\", default=0.8, type=float, help=\"Ratio of examples to use as train.\")
    args = parser.parse_args()

    # Create logdir name
    args.logdir = \"logs/{}-{}-{}\".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime(\"%Y-%m-%d_%H%M%S\"),
        \",\".join((\"{}={}\".format(re.sub(\"(.)[^_]*_?\", r\"\\1\", key), value) for key, value in sorted(vars(args).items())))
    )
    if not os.path.exists(\"logs\"): os.mkdir(\"logs\")  # TF 1.6 will do this by itself

    # Load the data
    train, dev = Dataset(\"modelnet{}-train.npz\".format(args.modelnet_dim)).split(args.train_split)
    test = Dataset(\"modelnet{}-test.npz\".format(args.modelnet_dim), shuffle_batches=False)

    # Construct the network
    network = Network(threads=args.threads)
    network.construct(args)

    # Train
    for i in range(args.epochs):
        while not train.epoch_finished():
            voxels, labels = train.next_batch(args.batch_size)
            network.train(voxels, labels)

        network.evaluate(\"dev\", dev.voxels, dev.labels)

    # Predict test data
    with open(\"{}/3d_recognition_test.txt\".format(args.logdir), \"w\") as test_file:
        while not test.epoch_finished():
            voxels, _ = test.next_batch(args.batch_size)
            labels = network.predict(voxels)

            for label in labels:
                print(label, file=test_file)
"""

test_data = b'{Wp48S^xk9=GL@E0stWa761SMbT8$j;0mJxj$Hs4gFQ{NIp$$7Aq+YKHC7>g?n;+=IYybz8E(;p?w>DUf#5xW!1#Fx{P*nCn!zGpT4aQ1Vmry<p#eBtPIK!)I3Ym&FrY@lwMlVS=i7Jw0p~Y$R`sMV?692TpuLN{Y(GjJPhost`LaI@2A+WSD+5652N_$zjfzTlF+yc>^pHE>&avPkrL|b^WoXNc`rRwf)`(ExgsBI!EH?{6Ybi7anu`!j{t$}h^ip)f?ULw^cDSvQ-ch;3!PQ+3$2xL%MU8$k>h>kkST<qSXkZFz$1r89r1iPGu^V1suA~!1EP`Sp5czz$U&Gpki~vVi9%zPcPv+6(nu0sw>Vw{q&Prj4-OALv3kh`ulo{&)ek!*`FQ?5*S+2rV!*h0644>H9NWKmfo=Nep@QP<k87T$z?+m-s#}na2y6iHtg7;_R&9@scJyoFyH-maoUf1t2d2Bld^~c0MMiuXThdiM>);xW6wc`8Hf`DMN>4vHybUYluSeR8;c($5Pc>53wE8_B$VZSJ~n7@pWxJ8Yw{WO$7%PrOI^JTf^I8Ij2>dqvUb}TKrV`D6&bNhN~Z)v9c17Qt<s6!x<&Y9K}>}NU}Uu*)%5lqXxN;H$w%mS(;;VbS#C=_Y^PQV6(t8^;2Ydh{9{`7C+D8j_b``Y+OUBNHeYQI^8$j2hk;((UL&&;65{M7EpVnY6&L+neXGnv>BCw5lE5K5Wk8oJ<W1KCPl;l|oGw~j9zyy0V6^=@|UuId2dDXMVc?FsbHz4}j>BSH4!x}PUh@#)!}2l@)HB(<Wmd%XtJ0=`rgi>OP8)XA2lE6k8qZ>lTaiJK|LcNrjSF)gAB*x)@f00000ADUoPFwQ|*00F86q!j=F?1t8BvBYQl0ssI200dcD'

if __name__ == "__main__":
    import base64
    import io
    import lzma
    import sys

    with io.BytesIO(base64.b85decode(test_data)) as lzma_data:
        with lzma.open(lzma_data, "r") as lzma_file:
            sys.stdout.buffer.write(lzma_file.read())
