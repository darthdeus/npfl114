#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    X = []

    # Load data distribution, each data point on a line
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")

            X.append(line)
            # TODO: process the line, aggregating using Python data structures

    X = np.array(X)

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. If required,
    # the NumPy array might be created after loading the model distribution.

    probs = {}

    # Load model distribution, each line `word \t probability`.
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")

            k, v = line.split("\t")
            v = float(v)

            probs[k] = v

    # TODO: Create a NumPy array containing the model distribution.
    model_dist = np.array(list(probs.values()))

    data_dist = np.unique(X, return_counts=True)[1].astype(np.float32) / len(X)

    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -np.sum(data_dist * np.log(data_dist))

    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    cross_entropy = -np.sum(data_dist * np.log(model_dist))
    kl_divergence = -np.sum(data_dist * np.log(model_dist / data_dist))

    print("{:.2f}".format(cross_entropy))
    print("{:.2f}".format(kl_divergence))

