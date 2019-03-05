#!/usr/bin/env python3
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="numpy_entropy_data.txt")
    parser.add_argument("--model", type=str, default="numpy_entropy_model.txt")

    args = parser.parse_args()

    X = []

    # Load data distribution, each data point on a line
    # with open("numpy_entropy_data.txt", "r") as data:
    with open(args.data, "r") as data:
        for line in data:
            line = line.rstrip("\n")

            X.append(line)
            # TODO: process the line, aggregating using Python data structures

    X = np.array(X)

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. If required,
    # the NumPy array might be created after loading the model distribution.

    # Load model distribution, each line `word \t probability`.
    model_dist_map = {}

    with open(args.model, "r") as model:
        for line in model:
            line = line.rstrip("\n")

            k, v = line.split("\t")
            v = float(v)

            model_dist_map[k] = v

    # TODO: Create a NumPy array containing the model distribution.
    data_words, data_counts = np.unique(X, return_counts=True)
    data_probs = data_counts.astype(np.float32) / len(X)

    data_dist_map = {}

    for word, prob in zip(data_words, data_probs):
        data_dist_map[word] = prob

    for key in model_dist_map.keys():
        if key not in data_dist_map:
            data_dist_map[key] = 0.0

    for key in data_dist_map.keys():
        if key not in model_dist_map:
            model_dist_map[key] = 0.0


    assert set(model_dist_map.keys()) == set(data_dist_map.keys())

    model_dist, data_dist = [], []

    for key in sorted(model_dist_map.keys()):
        model_dist.append(model_dist_map[key])
        data_dist.append(data_dist_map[key])

    model_dist = np.array(model_dist, dtype=np.float32)
    data_dist = np.array(data_dist, dtype=np.float32)

    # model_dist = np.array(list(model_dist_map.values()))
    # data_dist = np.array(list(data_dist_map.values()))

    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).

    nonzero_data_dist = data_dist[data_dist > 0]

    entropy = -np.sum(nonzero_data_dist * np.log(nonzero_data_dist))

    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)

    if np.any(model_dist == 0.0):
        cross_entropy = np.inf
        kl_divergence = np.inf
    else:
        cross_entropy = -np.sum(data_dist * np.log(model_dist))
        kl_divergence = -np.sum(data_dist * np.log(model_dist / data_dist))

    print("{:.2f}".format(cross_entropy))
    print("{:.2f}".format(kl_divergence))

