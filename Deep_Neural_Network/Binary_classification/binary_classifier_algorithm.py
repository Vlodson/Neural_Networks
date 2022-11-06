""" OVA SKRIPTA JE SAMO DA POKAZE KAKO RADI, MAKSIMALNO JE PROST, JEDAN SKRIVENI SLOJ SA 3 NEURONA I TO JE TO """

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

Data = Data[:100] # posto mi trebaju samo dve klase onda je prvih 100, da nisu sortirani vrv bih se ubio

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
y = np.array(y)
y = y.reshape((y.shape[0], 1))
#================

# potrebne funkcije

""" VELIKI PROBLEM AKO SE PROSLEDE VELIKE VREDNOSTI OVIMA DVEMA FUNKCIJAMA """
def sigmoid(value): # aktivaciona funkcija za izlazni sloj (skuplja brojeve na izmedju 0 i 1)
    return 1 / (1 + e**(-1 * value)) # formula je s(x) = 1 / (1 + e^-x)

def sigmoid_derivative(value): # izvod sigma funkcije, bice objasnjeno zasto kasnije
    return (1 - sigmoid(value)) * sigmoid(value) # formula je s'(x) = (1 - s(x)) * s(x) 

def tanh(value): # vraca tangens hiperbolicki za dati vektor, aktivaciona za hidden slojeve
    return np.tanh(value) # tanh znas

def tanh_derivative(value): # izvod hiperbolickog tangensa
    return 1 - np.tanh(value)**2 # t'(x) = 1 - t(x)^2

def transfer(inp, weights, bias): # vrsi transfer funkciju nad svim podacima za date tezine i biase
    return np.dot(inp, weights) + bias # z(x, w, b) = x kruzic w + b (kruzic je mnozenje dve matrice)

def MSE(y, yHat, n): # radi mean square error za prave vrednosti (y) i predvidjene vrednosti (yHat)
    sq = (y - yHat)**2
    return 1/(2*n)*np.sum(sq) # L(y, yHat) = 1/2n * suma ((y - yhat)^2)

#======

# pravljenje varijabli

input_neurons = X.shape[1]
hidden_neurons = 3
output_neurons = 1 # jedan neuron, jer je samo 0 ili 1
data_points = X.shape[0]

learn_rate = 1 # menjaj kad treba
learn_iter = 10000

# random inicijalizacija tezina i biasa (obrati paznju da moraju da budu jako mali, zato interval (0, 0.2) i (0, 0.3))
weights1 = np.random.uniform(0, 0.2, (input_neurons, hidden_neurons))
bias1 = np.random.uniform(0, 0.3, (1, hidden_neurons))

weights2 = np.random.uniform(0, 0.2, (hidden_neurons, output_neurons))
bias2 = np.random.uniform(0, 0.3, (1, output_neurons))

L = [] # trebace mi za grafik, u nju stavljam sve lossove da pratim na kraju

#======

# ovde je deo gde zapravo uci, sastoji se iz dva dela, forward i back propagation

i = 0
while i < learn_iter:

    # forward: sastoji se iz dva dela, prvi deo je prebacivanje vrednosti iz jednog u drugi sloj putem transfer funkcije koja mnozi (matricno) vrednosti proslog sloja sa tezinama i dodaje posle bias
    # drugi deo je aktivaciona funkcija, koja smanjuje vrednosti, ovde koriscenje tanh za skrivene slojeve i sigma za izlazni (sigma i MSE moraju da idu zajedno)
    # izlaz izlaznog sloja su predvidjanja mreze koja onda idu na Loss
    z1 = transfer(X, weights1, bias1)
    a1 = tanh(z1)

    z2 = transfer(a1, weights2, bias2)
    yHat = sigmoid(z2)

    #---
    # back: ovde gledas da minimizujes Loss funkciju preko hiperparametara (weights i biases) tako sto ces raditi parcijalne izvode loss-a po tezinama i po biasima
    # isto se sastoji iz dva glavna dela, racunanje izvoda za ilazni sloj i racunanje izlaza za skrivene
    L.append(MSE(y, yHat, data_points))

    # posto je Loss slozena funkcija, ovde sam rucno izvukao posebno delove, treba ti:
    dyHat = -1/data_points * (y - yHat) # izvod lossa po izlazu, tj. dL/dyHat sto je -1/n * (y - yHat)
    dz2 = dyHat * sigmoid_derivative(z2) # izvod transfera pre aktivacije, yHat je zapravo a(z), a a(z) je sigma pa: da/dz = da/ds * ds/dz = a*s'(z)
    # za oba ide 1/n zbog sumacije koja se vrsi u transferu
    dw2 = 1/data_points * np.dot(a1.T, dz2) # i na kraju izvod tezina po transferu i na kraju dz/dw = a^T kruzic dz, zbog matrica (otuda i transponovanje, zbog dimenzionalnosti), inace bi bilo samo puta
    db2 = 1/data_points* np.sum(dz2) # za bias je prosto jer dz/db je samo suma svih dz posto je +b samo sa strane 

    # dw i db za hidden slojeve, jedina razlika je u prvom da, sve ostalo je isto
    da1 = np.dot(dz2, weights2.T) # posto se ono sa izlaznog sloja propagira nazad, samo ides inverznom operacijom od transfer funkcije sto je matricno mnozenje sa transponovanim tegovima i greskama proslog sloja
    dz1 = da1 * tanh_derivative(z1)
    dw1 = 1/data_points * np.dot(X.T, dz1)
    db1 = 1/data_points * np.sum(dz1)
    
    # updateovanje tezine i biasa
    # novo = staro - ni * greska
    weights2 = weights2 - learn_rate*dw2 # w_novi = w_stari - ni*dW, ni je konstanta da se ne bi prebrzo spustao jer tangenta koju imas u tom trenutku ne mora da bude skroz dobra pa je dobro ici malo po malo
    bias2 = bias2 - learn_rate*db2
    weights1 = weights1 - learn_rate*dw1
    bias1 = bias1 - learn_rate*db1

    i += 1

plt.plot(L)
plt.show()