#!/usr/bin/env python3
import numpy as np

def shannon_entropy(P):
    P = P[P > 0]
    return -np.sum(P * np.log(P))

def cross_entropy(P, Q):
    mask = P > 0
    P, Q = P[mask], Q[mask]
    
    return -np.sum(P * np.log(Q))

def kl_divergence(P, Q):
    mask = P > 0
    P, Q = P[mask], Q[mask]
    
    return cross_entropy(P, Q) - shannon_entropy(P)

if __name__ == "__main__":
    data_items = []
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            data_items.append(line)
    
    data, counts = np.unique(data_items, return_counts=True)
    counts = counts.astype(np.float32)
    data_probs = counts / counts.sum()

    data_dist = dict(zip(data, data_probs))

    model_dist = {}
    
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            key, prob = line.rstrip("\n").split("\t")
            model_dist[key] = float(prob)

    for (key, value) in data_dist.items():
        if not key in model_dist.keys():
            model_dist[key] = 0.0
            
    for (key, value) in model_dist.items():
        if not key in data_dist.keys():
            data_dist[key] = 0.0
    
    P = [data_dist[key] for key in sorted(data_dist.keys())]
    P = np.fromiter(P, dtype=np.float32)
    Q = [model_dist[key] for key in sorted(model_dist.keys())]
    Q = np.fromiter(Q, dtype=np.float32)

    print("{:.2f}".format(shannon_entropy(P)))
    print("{:.2f}".format(cross_entropy(P, Q)))
    print("{:.2f}".format(kl_divergence(P, Q)))
