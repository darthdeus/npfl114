import pdb
import tensorflow as tf
import numpy as np

from glob import glob

import uppercase
from uppercase import Dataset, Network, parse_args

dirs = glob("models/*")

threads=12

nets = []

for dir in dirs:
    args, _ = parse_args()

    net = Network(threads=12)
    net.construct(args)
    net.load(dir)

    nets.append(net)

train = Dataset("data/uppercase_data_train.txt", args.window, alphabet=args.alphabet_size)
dev = Dataset("data/uppercase_data_dev.txt", args.window, alphabet=train.alphabet)
dev_windows, dev_labels = dev.all_data()

X, y = train.next_batch(64)

all_preds = np.array([net.predict(dev_windows) for net in nets])

ensemble_preds = all_preds.astype(np.float32).mean(axis=0).round().astype(np.bool)

isolated_acc = (dev_labels.reshape(1, -1) == all_preds).mean(axis=1)
ensemble_acc = (ensemble_preds == dev_labels).mean()

print("Mean acc: {:.5f}\tEnsemble acc: {:.5f}".format(isolated_acc.mean(), ensemble_acc))
print("Isolated acc: {}", isolated_acc)
pdb.set_trace()


# saver.restore(sess, )
