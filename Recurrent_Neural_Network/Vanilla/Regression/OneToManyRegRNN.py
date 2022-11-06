# pogeldati many to many reg za detaljnije, pisacu samo sta se razlikuje u odnosu na to

"""
Za razliku od many-many i many-one:
treba mu vise li
treba mu vise neurona u stanju (ovo mnogo vise utice na tacnost), a i ima smisla jer stanja nose sve
"""

import numpy as np
import matplotlib.pyplot as plt

data1d = np.arange(101, 1, -1, dtype='float64').reshape(100,1,1)
wn = np.random.uniform(-10, 10, data1d.shape)
data1d += wn
data1d = (data1d - np.min(data1d)) / (np.max(data1d) - np.min(data1d))

X = data1d[0].reshape(1, data1d.shape[1], data1d.shape[2]) # samo 1 X tj. timesteps Xa je 1
y = data1d[1:]
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

timesteps = y.shape[0] # y jer x ima samo 1 timepoint
datapoints = X.shape[1]
stateShape = 5 # mora da ima vise state neurona

zstatexh = np.zeros((timesteps, datapoints, stateShape))

zstatehh = np.zeros((timesteps, datapoints, stateShape))
dzstatehh = np.zeros_like(zstatehh)

zstate = np.zeros((timesteps, datapoints, stateShape))
dzstate = np.zeros_like(zstate)

state = np.zeros((timesteps + 1, datapoints, stateShape)) # t je za proslo, t+1 za trenutno, samo za njega

Loss = []
li = int(1e3)
lr = 1e-0

U = np.ones((1, X.shape[2], stateShape)) # timestep 1 jer samo 1 se racuna prelaz x u state
V = np.ones((timesteps, stateShape, y.shape[2]))
W = np.ones((stateShape, stateShape))

dW = 0
#==================

for i in range(li):
    dW = 0
    if i % (li/10) == 0:
        print("iter {} od {}".format(i,li))

    """ forward """
    zstatexh[0] = np.matmul(X, U) # samo 1 ovo i postoji pa zato [0]

    for t in range(timesteps): # mora, jebiga, zavisi trenutni od proslog
        zstatehh[t] = np.matmul(state[t], W)
        zstate[t] = zstatehh[t] + zstatexh[t]
        state[t+1] = tanh(zstate[t])

    zyh = np.matmul(state[1:], V) # od 1 jer se 0 ne vazi, vestacka je
    yh = sigmoid(zyh)

    Loss.append(MSE(y, yh))

    """ back """
    dyh = yh - y
    dzyh = dsigmoid(zyh) * dyh

    dV = np.matmul( np.transpose(state[1:], (0,2,1)), dzyh ) # numpy je prelep.

    dstate = np.matmul( dzyh, np.transpose(V, (0,2,1)) )
    dzstate = dtanh(zstate) * dstate

    dU = np.matmul( np.transpose(X, (0,2,1)), dzstate[0] ) # samo poslednje stanje igra ulogu u gradijentu U

    for t in range(timesteps-1, 0, -1): # mislim da ovo moze da se ubrza
        dW += np.matmul( state[t].T, dzstate[t] )

    """ update """
    U -= dU*lr
    W -= dW*lr
    V -= dV*lr

py = y.ravel()
pyh = yh.ravel()

fig, ax = plt.subplots(2, 1)
ax[0].plot(Loss)
ax[1].plot(py, 'b-', pyh, 'r-')
plt.show()