""" PRIMER RADA ALGORITMA NA MREZI OD JEDNOG SKRIVENOG SLOJA NA SKUPU IRIS """
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

X = []
y = []
ctr = 0
temp = [df.loc[0, 'Class']]

for dp in Data:
    X.append(dp[:4])

    if dp[4] not in temp:
        temp.append(dp[4])
        ctr += 1

    y.append(ctr)

X = np.array(X)

classes = ctr + 1 # kolko je brojac dosao toliko klasa ima (plus jedan jer krece od nula)


for i in range(len(y)):
    temp = np.zeros(classes)
    temp[y[i]] = 1
    y[i] = temp

y = np.array(y)

#==================================

# potrebne funkcije

def tanh(value): # vraca tangens hiperbolicki za dati vektor, aktivaciona za hidden slojeve
    return np.tanh(value) # tanh znas

def tanh_derivative(value): # izvod hiperbolickog tangensa
    return 1 - np.tanh(value)**2 # t'(x) = 1 - t(x)^2

# def transfer(inp, weights, bias): # vrsi transfer funkciju nad svim podacima za date tezine i biase
#     return np.dot(weights.T, inp.T).T + bias # z(x, w, b) = x kruzic w + b (kruzic je mnozenje dve matrice)

def transfer(inp, weights, bias):
    return np.dot(inp, weights) + bias

def CE_Loss(y, yHat):
    return -1 * np.sum( y * np.log(yHat) )

def softmax(matrix):
    mx = np.max(matrix, axis = 1)
    mx = mx.reshape( mx.shape[0], 1 )
    
    exp = np.exp(matrix - mx)
    sm = np.sum(exp, axis = 1)

    return (exp.T / sm).T

#======

# potrebne promenljive 

input_neurons = X.shape[1]
hidden_neurons = 3
output_neurons = classes
data_points = X.shape[0]

learn_rate = 5e-3
epochs = int(1e5)

w1 = np.random.uniform(0, 0.2, (input_neurons, hidden_neurons))
b1 = np.random.uniform(0, 0.3, (1, hidden_neurons))

w2 = np.random.uniform(0, 0.2, (hidden_neurons, output_neurons))
b2 = np.random.uniform(0, 0.3, (1, output_neurons))

L = []

for i in range(epochs):
    if i % int(epochs/10) == 0:
        print("Epoha {} od {}".format(i, epochs))

    z1 = transfer(X, w1, b1)
    a1 = tanh(z1)

    z2 = transfer(a1, w2, b2)
    a2 = softmax(z2)

    L.append(CE_Loss(y, a2))
    #---

    dz2 = a2 - y
    dw2 = np.dot( a1.T, dz2 ) / data_points
    db2 = np.sum(dz2) / data_points

    da1 = np.dot(dz2, w2.T)
    dz1 = da1 * tanh_derivative(z1)
    dw1 = np.dot( X.T, dz1 ) / data_points
    db1 = np.sum(dz1) / data_points

    w2 = w2 - learn_rate*dw2
    w1 = w1 - learn_rate*dw1
    b2 = b2 - learn_rate*db2
    b1 = b1 - learn_rate*db1

    i += 1

plt.plot(L)
plt.show()