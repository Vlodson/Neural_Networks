import pandas as pd
import numpy as np

# read and drop nans
df = pd.read_csv("penguins_size.csv")
df.dropna(axis=0, how='any', inplace=True)

# seperate input and labels
X = df[['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']]
y = df['species']

# turn to array and normalize
X = X.to_numpy(dtype=np.float_)
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# one hot encode y and reshape it to be a vector
y = y.to_numpy(dtype=np.object_)
penguin_types = np.unique(y)

labels = np.zeros((y.shape[0], penguin_types.shape[0]))
for penguin in penguin_types:
    labels[y == penguin, penguin_types == penguin] = 1

y = np.array(labels, dtype=np.float_)

# shuffle both X and y
idx = np.arange(X.shape[0])
np.random.shuffle(idx)

X = X[idx]
y = y[idx]