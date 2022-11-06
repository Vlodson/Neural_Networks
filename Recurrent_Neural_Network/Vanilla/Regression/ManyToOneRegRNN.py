# pogeldati many to many reg za detaljnije, pisacu samo sta se razlikuje u odnosu na to

import numpy as np
import matplotlib.pyplot as plt

# data1d = np.arange(1, 6).reshape(5,1,1)
data1d = np.random.randint(0, 100, (100, 1, 1))
data1d = data1d / np.max(data1d)

X = data1d[:-1]
y = data1d[-1].reshape(1, data1d.shape[1], data1d.shape[2]) # reshape da bude 3D matrica sa 1 timestepom
#==================

def tanh(data):
    return np.tanh(data)

def dtanh(data):
    return 1 - tanh(data)**2

def sigmoid(data):
    return np.where(data > 0, 1 / (1 + np.exp(-data)), np.exp(data) / (1 + np.exp(data)))

def dsigmoid(data):
    return sigmoid(data) * (1 - sigmoid(data))

def MSE(y, yh): # vratice za celu iteraciju ucenja loss
    return .5 * np.sum((y - yh)**2) / y.shape[1]
#==================

timesteps = X.shape[0]
datapoints = X.shape[1]
stateShape = 3

zstatehh = np.zeros((timesteps, datapoints, stateShape))
dzstatehh = np.zeros_like(zstatehh)

zstate = np.zeros((timesteps, datapoints, stateShape))
dzstate = np.zeros_like(zstate)

state = np.zeros((timesteps + 1, datapoints, stateShape)) # t je za proslo, t+1 za trenutno, samo za njega

Loss = []
li = int(5e2)
lr = 1e-0

U = np.ones((timesteps, X.shape[2], stateShape))
V = np.ones((1, stateShape, y.shape[2])) # axis(0) = 1 jer samo u poslednjem timestepu se racuna izlaz, tj timesteps za V je samo 1
W = np.ones((stateShape, stateShape))

dW = 0
#==================

for i in range(li):
    dW = 0
    if i % (li/10) == 0:
        print("iter {} od {}".format(i,li))

    """ forward """
    zstatexh = np.matmul(X, U)

    for t in range(timesteps):
        zstatehh[t] = np.matmul(state[t], W)
        zstate[t] = zstatehh[t] + zstatexh[t]
        state[t+1] = tanh(zstate[t])

    zyh = np.matmul(state[-1], V) # samo poslednji state mnozim sa V jer samo u njemu racunam izlaz
    yh = sigmoid(zyh)

    Loss.append(MSE(y, yh))

    """ back """
    dyh = yh - y
    dzyh = dsigmoid(zyh) * dyh

    dV = np.matmul( np.transpose(state[-1]), dzyh ) # opet samo poslednji state igra ulogu u dV

    dstate = np.matmul( dzyh, np.transpose(V, (0,2,1)) )
    dzstate = dtanh(zstate) * dstate

    dU = np.matmul( np.transpose(X, (0,2,1)), dzstate )

    for t in range(timesteps-1, 0, -1):
        dW += np.matmul( state[t].T, dzstate[t] )

    """ update """
    U -= dU*lr
    W -= dW*lr
    V -= dV*lr
#==================

plt.plot(Loss)
plt.show()