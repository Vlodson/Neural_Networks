from typing import List, Union
from neural_network_utils import *
import decoder as dec
from size_dataset import X, y
import matplotlib.pyplot as plt


def train(X_enc, X_dec, y_dec, learn_rate, epochs, verbose):
    # since it's an autoencoder, it only has Xs, which are actually labels y
    w1, b1 = create_layer_params((3,2))
    w2, b2 = create_layer_params((2,4))

    print("Training decoder...")
    w, b = dec.train(X_dec, y_dec, 1e0, int(1e3), False)
    wdec1, wdec2 = w
    bdec1, bdec2 = b
    print("Decoder trained\n")
    
    print("Training encoder...")
    L = []
    for i in range(epochs):
        if i % (epochs // 10) == 0:
            print(f"Epoch {i} of {epochs}")

        # forward
        z1 = transfer(X_enc, w1, b1)
        a1 = activate(z1, tanh)

        z2 = transfer(a1, w2, b2)
        a2 = activate(z2, sigmoid)

        # trained forward
        zdec1 = transfer(a2, wdec1, bdec1)
        adec1 = activate(zdec1, tanh)

        zdec2 = transfer(adec1, wdec2, bdec2)
        adec2 = activate(zdec2, softmax)

        # overall Loss
        L.append(calculate_loss(adec2, y))

        # propagate error to encoder
        dzdec2 = (adec2 - X_enc) / X_enc.shape[0]
        _, _, dadec1 = delta(dzdec2, None, adec1, wdec2, zdec2, False)
        _, _, dX = delta(dadec1, dtanh, a2, wdec1, bdec1, True)

        # backpropagate through encoder
        dw2, db2, da1 = delta(dX, dsigmoid, a1, w2, z2, True)
        dw1, db1, _ = delta(da1, dtanh, X_enc, w1, z1, True)

        # update encoder
        w2 = update(w2, dw2, learn_rate)
        b2 = update(b2, db2, learn_rate)

        w1 = update(w1, dw1, learn_rate)
        b1 = update(b1, db1, learn_rate)
    print("Encoder training finished")

    if verbose:
        plt.plot(L)
        plt.show()

    return [w1, w2], [b1, b2]


if __name__ == '__main__':
    w, b = train(y, X, y, 1e-1, int(1e3), True)