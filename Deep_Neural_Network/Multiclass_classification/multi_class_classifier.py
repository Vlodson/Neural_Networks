""" ZA SADA JE NAJVAZNIJE SAMO DA RADI KOLKO TOLKO, POSLE GENERALIZUJ SVE U FUNKCIJU JEDNU ILI KLASU """

import numpy as np
import pandas as pd
from math import e
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/vlada/Desktop/Datasets/Iris-flowers/Dataset.csv")

# Raspakivam dataframe u listu
Data = []

for row in range(len(df)):
    temp = []
    for collumn in df:
        temp.append(df.loc[row, collumn])
    Data.append(temp)

#======

# Pravim input vektor i labele
X = []
y = []
temp = [df.loc[0, 'Class']]
ctr = 0

# na X dodajem sirove podatke, a na y ravim 1 hot-encoder tako sto za svaku labelu na mestu brojaca stavim 1, ako je klasa drugacija pomeri se brojac
for dp in Data:
    label = [0,0,0] # mrzi me sada da pravim generalizovano, znam da iris ima 3 klase
    X.append(dp[:4])

    if dp[4] not in temp: # ovde appendujem novu klasu jer ne mora df da bude sortiran
        ctr += 1
        temp.append(dp[4])

    label[ctr] = 1
    y.append(label)

#======

X = np.array(X)
y = np.array(y)

# potrebne funkcije

""" VELIKI PROBLEM AKO SE PROSLEDE VELIKE VREDNOSTI OVIMA DVEMA FUNKCIJAMA """
def sigmoid(vector): # dajes Z
    return 1 / (1 + e**(-1 * vector))

def sigmoid_derivative(vector): # ovde mora Z da udje
    return (1 - sigmoid(vector)) * sigmoid(vector)

def tanh(vector): # vraca tangens hiperbolicki za dati vektor
    return np.tanh(vector)

def tanh_derivative(vector): # ovde vektor koji prosledjujes je transferovani vektor
    return 1 - np.tanh(vector)**2

#---

def transfer(inp, weights, bias): # ovde prosledjujes sve inpute
    return np.dot(inp, weights) + bias

def MSE(y, yHat, n): # ovde pojedinacne vektore. Za cost sumiraj sve njih
    return  1/(2*n)*np.sum((y - yHat)**2)

def dZ(dA, df): # dZ ovog sloja, dA se uzima i uzima se izvod aktivacione za taj sloj
    return dA * df

def dA(W, dZ): # dA za prethodni sloj, uzimas weights izmedju slojeva i dZ ovog sloja
    return np.dot(dZ, W.T)

def db(dZ, m): # db je suma svih dZ
    return 1/m * np.sum(dZ) # malo fishy

def dW(dZ, A): # uzima dZ ovog sloja i A prethodnog
    return np.dot(A.T, dZ)

#======

# pravljenje varijabli

input_neurons = X.shape[1]
hidden_neurons = 3
output_neurons = y.shape[1]
data_points = X.shape[0]

learn_rate = 0.1 # menjaj kad treba
learn_iter = 5

weights1 = np.random.uniform(0, 0.1, (input_neurons, hidden_neurons))
bias1 = np.random.uniform(0, 0.3, (1, hidden_neurons))

weights2 = np.random.uniform(0, 0.1, (hidden_neurons, output_neurons))
bias2 = np.random.uniform(0, 0.3, (1, output_neurons))

L = []

#======

ctr = 0
while ctr < learn_iter:

    t01 = transfer(X, weights1, bias1)
    print(t01[0], '\n')

    a1 = tanh(t01)
    print(a1[0], '\n')

    t12 = transfer(a1, weights2, bias2)
    print(t12[0], '\n')

    a2 = sigmoid(t12)
    print(a2[0], '\n')
    #---

    L.append(MSE(y, a2, data_points))

    dA2 = 1/data_points * (a2 - y)
    print(dA2[0], '\n')
    dZ2 = dZ(dA2, sigmoid_derivative(t12))
    print(dZ2[0], '\n')
    db2 = db(dZ2, data_points)
    print(db2, '\n')
    dW2 = dW(dZ2, a1)
    print(dW2[0], '\n')

    dA1 = dA(weights2, dZ2)
    dZ1 = dZ(dA1, tanh_derivative(t01))
    db1 = db(dZ1, data_points)
    dW1 = dW(dZ1, X)

    weights1 = weights1 - learn_iter * dW1
    bias1 = bias1 - learn_iter * db1
    weights2 = weights2 - learn_iter * dW2
    bias2 = bias2 - learn_iter * db2

    ctr += 1
    print('===========\n')


plt.plot(L)
plt.show()