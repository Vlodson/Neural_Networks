# many to many, sa outputom za svaki korak

# ulaz da bude oblika (timesteps, datapoints, features), a posto je regresija, datapoints = 1
# U da bude (timesteps, infeatures, statefeatures)
# V da bude (timesteps, statefeatures, outfeatures)
# W da bude (statefeatures, statefeatures)

# np.matmul(X, U) daje (timesteps, datapoints, statefeatures)

# transponovanje: np.transpose(X, (0, 2, 1)) ovaj tuple su axes, 0 ostaje a axis 2 i 1 su se zamenili

# mislim da ne zna ako je vise datapointova, ali to ima smisla jer je ovo regresija, a ne clf. konvergira ka nekom proseku svih datapointova, sto i ima smisla

"""
mozes da izracunas sve X*U odma, onda mora for loop za h*W pa onda mozes opet matmul za sve izlaze odma 
"""
#==================
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# data2d = np.zeros((100, 1, 2))
# temp = np.arange(1, 101).reshape(100,1,1)
# data2d = data2d + temp # linearno rastuce tacke u obe dimenzije
# data2d = data2d / np.max(data2d, axis = 0) # norm

# data1d = np.arange(1, 11, dtype='float64').reshape(10, 1, 1)
# # data1d = np.random.uniform(-10, 10, (10, 1, 1))
# wn = np.random.uniform(-1, 1, data1d.shape)
# data1d += wn
# data1d = (data1d - np.min(data1d, axis = 0)) / (np.max(data1d, axis = 0) - np.min(data1d, axis = 0))

# X = data1d[:-1]
# y = data1d[1:]

df = pd.read_csv("C:/Users/vlada/Desktop/Datasets/Soap-Sales/data.csv")

df = np.array(df)

t = df[:, :2]
data = df[:, -1]

data = (data - np.min(data)) / (np.max(data) - np.min(data)) # normalizacija

X = data[:-1]
y = data[1:]

X = X.reshape(X.shape[0], 1, 1)
y = y.reshape(y.shape[0], 1, 1)
#==================
""" aktivacione fukcije, njihovi izvodi i Loss """
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

""" init hiperparametara i init parametara """
timesteps = X.shape[0]
datapoints = X.shape[1]
stateShape = 3

zstatehh = np.zeros((timesteps, datapoints, stateShape)) # neaktivirani prelaz iz s(t-1) u s(t)
dzstatehh = np.zeros_like(zstatehh) # tensor za izvod njih

zstate = np.zeros((timesteps, datapoints, stateShape)) # tensor neaktiviranog stanja h
dzstate = np.zeros_like(zstate) # tensoz za izvod stanja

# tensor stanja, t+1 u timesteps jer mora da ima za t=0 prethodno stanje, nema tensor za izvod jer se moze odma izracunati preko matmul
state = np.zeros((timesteps + 1, datapoints, stateShape)) # t je za proslo, t+1 za trenutno, samo za njega

Loss = []
li = int(5e2)
lr = 1e-0

# init tezina, oblici objasnjeni u Doc.txt
U = np.ones((timesteps, X.shape[2], stateShape))
V = np.ones((timesteps, stateShape, y.shape[2]))
W = np.ones((stateShape, stateShape))

dW = 0
#==================

for i in range(li):
    dW = 0
    if i % (li/10) == 0:
        print("iter {} od {}".format(i,li))

    """ forward """
    zstatexh = np.matmul(X, U) # racunanje svih prelaza x u skriveno stanje (xh znaci iz x u h)

    # racunanje stanja
    for t in range(timesteps): # mora, jebiga, zavisi trenutni od proslog
        zstatehh[t] = np.matmul(state[t], W) # prelaz h u h
        zstate[t] = zstatehh[t] + zstatexh[t] # neaktivirano trenutno stanje
        state[t+1] = tanh(zstate[t]) # aktivirano trenutno stanje

    zyh = np.matmul(state[1:], V) # od 1 jer se 0 ne vazi, vestacka je
    yh = sigmoid(zyh)

    Loss.append(MSE(y, yh))

    """ back """
    dyh = yh - y # dL/dyH
    dzyh = dsigmoid(zyh) * dyh # dyH/dzyH

    # dzyh/dV
    dV = np.matmul( np.transpose(state[1:], (0,2,1)), dzyh ) # numpy je prelep.

    dstate = np.matmul( dzyh, np.transpose(V, (0,2,1)) ) # dzyh/ds
    dzstate = dtanh(zstate) * dstate # ds / dzs

    dU = np.matmul( np.transpose(X, (0,2,1)), dzstate ) # dzs/dU

    # dzs/dW kroz vreme
    for t in range(timesteps-1, 0, -1): # mislim da ovo moze da se ubrza
        dW += np.matmul( state[t].T, dzstate[t] )

    """ update """
    U -= dU*lr
    W -= dW*lr
    V -= dV*lr
#==================

py = y.reshape(y.shape[0])
pyh = yh.reshape(yh.shape[0])

fig, ax = plt.subplots(2, 1)
ax[0].plot(Loss)
ax[1].plot(py, 'b-', pyh, 'r-')
#==================