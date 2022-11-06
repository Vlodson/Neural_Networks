from neural_network_utils import *
import encoder as enc
from size_dataset import X, y
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    w, b = enc.train(y, X, y, 1e-1, int(1e3), False)

    wenc1, wenc2 = w
    benc1, benc2 = b

    Xtest = np.array([0., 1., 0.]).reshape(1, -1)
    z1 = transfer(Xtest, wenc1, benc1)
    a1 = activate(z1, tanh)

    z2 = transfer(a1, wenc2, benc2)
    a2 = activate(z2, sigmoid)

    realX = X[y[:, 1] == 1].mean(axis=0)

    print(a2)
    print(realX)