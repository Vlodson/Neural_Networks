# pogeldati many to one reg

"""
-ucenje je jako veliki zent, bas jako veliki, osciluje jako mnogo,
ali mora kako bi usao u bas GLOBALNI minimum, jer lokalni vodi do 0.5 0.5 verovatnoca
-li 1e3 lr 1e-2 i nekoliko pokretanja dovode do globalnog minimuma, na ts 5 dp 20 v 1 
-ts 5 dp 200 li 1e4 lr 1e-4 v 1 i nekoliko pokretanja = Loss 0
-ts 10 dp 20 li 1e3 lr 1e-2 v 1 Loss 0
-ts 10 dp 200 li 1e4 lr 1e-4 v 1 Loss 0
-ts 100 baca NaN za logaritam
-vanishing gradient na ts 100 (verovatno i ranije) (?)
"""

import numpy as np
import matplotlib.pyplot as plt
import random as rnd

ts = 5
dp = 20
v = 1

data = np.arange(1, 2*ts+1, dtype='float64').reshape(2*ts, 1, 1) # sve moguce vrednosti
# data = np.array([5,4,3,2,1,6,7,8,9,10]).reshape(2*ts, 1, 1)
c1 = data[:ts].reshape(ts, 1) # vrednosti klase 1
c2 = data[ts:].reshape(ts, 1) # vrednosti klase 2
ds = np.zeros((ts, dp, 1)) # dataset koji ce se dobiti sa sumom

for i in range( int(ds.shape[1] / 2) ): # pravljenje ds
    wn1 = np.random.uniform(-v, v, c1.shape) # sumovi
    wn2 = np.random.uniform(-v, v, c2.shape)

    ds[:, i, :] = c1 + wn1
    ds[:, ds.shape[1] - i - 1, :] = c2 + wn2

X = (ds - np.min(ds)) / (np.max(ds) - np.min(ds)) # norm

y = np.zeros((dp, 2)) # labele
y[:int(dp/2), 0] = 1
y[int(dp/2):, 1] = 1

idx = rnd.sample( list(range(dp)), dp ) # shuffle
X = X[:, idx, :]
y = y[idx].reshape(1, y.shape[0], y.shape[1])
#==================

def tanh(data):
    return np.tanh(data)

def dtanh(data):
    return 1 - tanh(data)**2

def ReLU(data):
    return data * (data > 0)

def dReLU(data):
    return 1 * (data > 0)

def softmax(data): # (t, d, f)
    mx = np.max(data, axis = 2) # malo drugaciji softmax zbog 3d ulaza u sm, pa je axis=2 svuda gde je bilo axis=1
    mx = mx.reshape(mx.shape[0], mx.shape[1], 1)

    numerator = np.exp(data - mx)
    denominator = np.sum(numerator, axis = 2)
    denominator = denominator.reshape(denominator.shape[0], denominator.shape[1], 1)

    return numerator / denominator

def CELoss(y, yh):
    return -1 * np.sum( y * np.log(yh) )
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
li = int(1e3)
lr = 1e-3

U = np.random.uniform(-10, 10, (timesteps, X.shape[2], stateShape))*.1
V = np.random.uniform(-10, 10, (1, stateShape, y.shape[2]))*.1
W = np.random.uniform(-10, 10, (stateShape, stateShape))*.1

# U = np.ones((timesteps, X.shape[2], stateShape))
# V = np.ones((1, stateShape, y.shape[2]))
# W = np.ones((stateShape, stateShape))

dW = 0
#==================

for i in range(li):
    dW = 0
    if i % (li/10) == 0:
        print("iter {} od {}".format(i,li))

    zstatexh = np.matmul(X, U)

    for t in range(timesteps): # mora, jebiga, zavisi trenutni od proslog
        zstatehh[t] = np.matmul(state[t], W)
        zstate[t] = zstatehh[t] + zstatexh[t]
        state[t+1] = ReLU(zstate[t])

    zyh = np.matmul(state[-1], V) # od 1 jer se 0 ne vazi, vestacka je
    yh = softmax(zyh)

    Loss.append(CELoss(y, yh))

    dzyh = yh - y
    dV = np.matmul( np.transpose(state[-1]), dzyh )

    dstate = np.matmul( dzyh, np.transpose(V, (0,2,1)) )
    dzstate = dReLU(zstate) * dstate

    dU = np.matmul( np.transpose(X, (0,2,1)), dzstate )

    for t in range(timesteps-1, 0, -1):
        dW += np.matmul( state[t].T, dzstate[t] )

    U -= dU*lr
    W -= dW*lr
    V -= dV*lr
#==================

plt.plot(Loss)
plt.show()
#==================

# validacija
zs = np.zeros((timesteps, 1, stateShape))
s = np.zeros((timesteps+1, 1, stateShape))

zsxh = np.matmul(c1.reshape(c1.shape[0], 1, c1.shape[1]), U)
for t in range(timesteps):
    zs[t] = np.matmul(s[t], W) + zsxh[t]
    s[t+1] = ReLU(zs[t])

zyht = np.matmul(s[-1], V)
yht = softmax(zyht)