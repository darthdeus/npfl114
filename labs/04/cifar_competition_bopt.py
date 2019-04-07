#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from cifar10 import CIFAR10

# The neural network model
class Network(tf.keras.Model):
    def __init__(self, args, cifar):
        inputs = tf.keras.layers.Input(shape=[CIFAR10.H, CIFAR10.W, CIFAR10.C])

        # TODO: Add CNN layers specified by `args.cnn`, which contains
        # comma-separated list of the following layers:
        # - `C-filters-kernel_size-stride-padding`: Add a convolutional layer with ReLU
        #   activation and specified number of filters, kernel size, stride and padding.
        # - `CB-filters-kernel_size-stride-padding`: Same as `C`, but use batch normalization.
        #   In detail, start with a convolutional layer without bias and activation,
        #   then add batch normalization layer, and finally ReLU activation.
        # - `M-kernel_size-stride`: Add max pooling with specified size and stride.
        # - `R-[layers]`: Add a residual connection. The `layers` contain a specification
        #   of at least one convolutional layer (but not a recursive residual connection `R`).
        #   The input to the specified layers is then added to their output.
        # - `F`: Flatten inputs. Must appear exactly once in the architecture.
        # - `D-hidden_layer_size`: Add a dense layer with ReLU activation and specified size.
        # Produce the results in variable `hidden`.
        hidden = inputs

        def conv_block(x, filters):
            x = layers.Conv2D(filters, 3, 1, "same", activation=None, use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)

            return x

        def resnet_block(input_x, filters):
            residual = layers.Conv2D(filters, 1, 1, "same")(input_x)

            x = input_x
            x = conv_block(x, filters)
            x = conv_block(x, filters)
            x = layers.Add()([x, residual])
            x = layers.MaxPool2D(2)(x)

            return x

        hidden = resnet_block(hidden, 64)
        hidden = resnet_block(hidden, 128)
        hidden = resnet_block(hidden, 256)

        hidden = layers.Flatten()(hidden)

        hidden = layers.Dense(1024, activation="relu")(hidden)
        hidden = layers.Dropout(args.dropout)(hidden)

        # Add the final output layer
        outputs = tf.keras.layers.Dense(cifar.LABELS, activation=tf.nn.softmax)(hidden)

        super().__init__(inputs=inputs, outputs=outputs)

        if args.learning_rate_final > args.learning_rate:
            args.learning_rate_final = args.learning_rate

        decay_steps = cifar.train.size * args.epochs // args.batch_size

        if args.decay == "polynomial":
            lr = tf.keras.optimizers.schedules.PolynomialDecay(args.learning_rate,
                    decay_steps, args.learning_rate_final)
        elif args.decay == "exponential":
            lr = tf.keras.optimizers.schedules.ExponentialDecay(args.learning_rate,
                    decay_steps, args.learning_rate_final / args.learning_rate)
        else:
            raise NotImplementedError()


        if args.optimizer == "SGD":
            optimizer = tf.keras.optimizers.SGD(lr, momentum=(args.momentum or 0.0))
        elif args.optimizer == "Adam":
            optimizer = tf.keras.optimizers.Adam(lr)
        else:
            raise NotImplementedError()

        print("ARGS: {}".format(args))

        self.compile(
            # optimizer=tf.keras.optimizers.Adam(args.learning_rate),
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

        self.tb_callback=tf.keras.callbacks.TensorBoard(args.logdir, update_freq=1000, profile_batch=1)
        self.tb_callback.on_train_end = lambda *_: None


    def train(self, cifar, args):
        self.fit(
            cifar.train.data["images"], cifar.train.data["labels"],
            batch_size=args.batch_size, epochs=args.epochs,
            validation_data=(cifar.dev.data["images"], cifar.dev.data["labels"]),
            callbacks=[self.tb_callback],
        )


if __name__ == "__main__":
    import argparse
    import datetime
    import os
    import re

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout rate.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="Learning rate.")
    parser.add_argument("--learning_rate_final", default=1e-6, type=float, help="Final learning rate.")
    parser.add_argument("--decay", default="exponential", type=str, help="Decay types.")
    parser.add_argument("--optimizer", default="SGD", type=str, help="Optimizer type.")
    parser.add_argument("--momentum", default=0.001, type=float, help="SGD momentum")

    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of epochs.")

    parser.add_argument("--cnn", default=None, type=str, help="Configuration string for the CNN.")
    parser.add_argument("--threads", default=11, type=int, help="Maximum number of threads to use.")
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

    # Load data
    cifar = CIFAR10()

    # Create the network and train
    network = Network(args, cifar)
    network.train(cifar, args)

    # Generate test set annotations, but in args.logdir to allow parallel execution.
    with open(os.path.join(args.logdir, "cifar_competition_test.txt"), "w", encoding="utf-8") as out_file:
        for probs in network.predict(cifar.test.data["images"], batch_size=args.batch_size):
            print(np.argmax(probs), file=out_file)

    print("RESULT={}".format(network.evaluate(cifar.dev.data["images"], cifar.dev.data["labels"])[1]))
