import numpy as np
import pandas as pd
from math import e
import matplotlib.pyplot as plt
import random as rnd

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

#========================

epochs = 5
batch_size = 64

batch = X
batch_l = y
ctr = 0

for i in range(epochs):
    print("epoha {}\n".format(i))

    while batch_size <= batch.shape[0]:
        idx = rnd.sample(range(0, batch.shape[0]), batch_size)
        mini_batch = batch[idx]
        mini_batch_l = batch_l[idx]
        ctr += 1
        print("uzeo mini batch {}".format(ctr))

        batch = np.delete(batch, idx, axis = 0)
        batch_l = np.delete(batch_l, idx, axis = 0)
        print("obrisao minibatch\n")

    if batch_size > batch.shape[0] and batch.shape[0] != 0:
        mini_batch = batch
        mini_batch_l = batch_l
        print("uzeo poslednji batch")

    batch = X
    batch_l = y
    ctr = 0
    print("restartovao batch\n\n")