from typing import Callable, Tuple
import numpy as np


def tanh(X: np.ndarray) -> np.ndarray:
    return np.tanh(X)


def dtanh(X: np.ndarray) -> np.ndarray:
    return 1 - tanh(X)**2


def sigmoid(X: np.ndarray) -> np.ndarray:
    return np.where(X >= 0, 1 / (1 + np.exp(-X)), np.exp(X) / (1 + np.exp(X)))


def dsigmoid(X: np.ndarray) -> np.ndarray:
    return sigmoid(X) * (1 - sigmoid(X))


def softmax(X: np.ndarray) -> np.ndarray:
    mxs = np.max(X, axis = 1).reshape(-1, 1)
    exps = np.exp(X - mxs)

    sums = np.sum(exps, axis = 1).reshape(-1, 1)

    return exps / sums


def create_layer_params(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    returns random w, b
    """
    return np.random.uniform(-10, 10, shape)*1e-1, np.random.uniform(-10, 10, (shape[1]))*1e-1


def transfer(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    returns z
    """
    return X @ W + b


def activate(X: np.ndarray, f: Callable) -> np.ndarray:
    """
    returns a
    """
    return f(X)


def calculate_loss(yh: np.ndarray, y: np.ndarray) -> float:
    """
    returns CE loss
    """
    return -1.*np.sum(y * np.log(yh)) / y.shape[0]


def delta(dy: np.ndarray, df: Callable, X: np.ndarray, W: np.ndarray, z: np.ndarray, deactivate: bool = True):
    """
    returns dw, db, dx
    """
    if deactivate:
        dz = dy * df(z)
    else:
        dz = dy
    dw = X.T @ dz
    db = np.sum(dz, axis=0)
    dX = dz @ W.T

    return dw, db, dX

def update(param: np.ndarray, grad: np.ndarray, eps: float) -> np.ndarray:
    """
    returns updated param using GD
    """
    updated_param = param - eps*grad
    return updated_param
