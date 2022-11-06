from typing import List, Union
from neural_network_utils import *
from size_dataset import X, y
import matplotlib.pyplot as plt


def train(X, y, learn_rate, epochs, verbose) -> Union[List[np.ndarray], List[np.ndarray]]:
    w1, b1 = create_layer_params((4,2))
    w2, b2 = create_layer_params((2,3))

    L = []
    for i in range(epochs):
        if i % (epochs // 10) == 0:
            print(f"Epoch {i} of {epochs}")

        # forward
        z1 = transfer(X, w1, b1)
        a1 = activate(z1, tanh)

        z2 = transfer(a1, w2, b2)
        a2 = activate(z2, softmax)

        L.append(calculate_loss(a2, y))

        # back
        dz2 = (a2 - y) / y.shape[0]
        dw2, db2, da1 = delta(dz2, None, a1, w2, z2, False)
        dw1, db1, _ = delta(da1, dtanh, X, w1, z1)

        # update
        w2 = update(w2, dw2, learn_rate)
        b2 = update(b2, db2, learn_rate)

        w1 = update(w1, dw1, learn_rate)
        b1 = update(b1, db1, learn_rate)

    if verbose:
        plt.plot(L)
        plt.show()

    return [w1, w2], [b1, b2]


if __name__ == '__main__':
    w, b = train(X, y, 1e0, int(1e3), True)