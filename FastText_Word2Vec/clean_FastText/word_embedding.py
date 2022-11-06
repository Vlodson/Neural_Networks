import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import List, Tuple, Dict
from operator import itemgetter

from global_cfg import EMBEDDING_CFG
from text_preprocessing import word_ngram


def _tanh(tensor: np.ndarray) -> np.ndarray:
    return np.tanh(tensor)


def _dtanh(tensor: np.ndarray) -> np.ndarray:
    return 1 - _tanh(tensor) ** 2


def _softmax(tensor: np.ndarray) -> np.ndarray:
    mxs = np.max(tensor, axis=1).reshape(-1, 1)
    exps = np.exp(tensor - mxs)

    sums = np.sum(exps, axis=1).reshape(-1, 1)

    return exps / sums


class WordEmbedding:
    def __init__(self, dataset: Dict[str, List[np.ndarray]], ngram_one_hots: Dict[str, np.ndarray]):
        self.data: List[np.ndarray] = [dataset[token][0] for token in dataset.keys()]
        self.labels: List[np.ndarray] = [dataset[token][1] for token in dataset.keys()]
        self.unique_tokens: Dict[str, np.ndarray] = {token: dataset[token][0] for token in
                                                     np.unique(list(dataset.keys()))}
        self.ngram_one_hots: Dict[str, np.ndarray] = ngram_one_hots
        self.input_shape = self.labels[0].shape[-1]

        self.embedding_space = np.random.uniform(-1, 1, size=(self.input_shape, EMBEDDING_CFG["embedding_size"]))
        self.weights = np.random.uniform(-1, 1, size=(EMBEDDING_CFG["embedding_size"], self.input_shape))

        self.loss = []

    def __forward(self, datapoint: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        transfer = datapoint @ self.embedding_space
        activation = _tanh(transfer)
        y_transfer = activation @ self.weights
        y_hat = _softmax(y_transfer)

        return transfer, activation, y_transfer, y_hat

    @staticmethod
    def __calculate_loss(y_hat: np.ndarray, label: np.ndarray) -> float:
        return -np.sum(label * y_hat / y_hat.shape[0])

    def __backprop(self, datapoint: np.ndarray, label: np.ndarray, transfer: np.ndarray, activation: np.ndarray,
                   y_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d_yz = (y_hat - label) / y_hat.shape[0]
        d_weights = activation.T @ d_yz
        d_activation = d_yz @ self.weights.T
        d_transfer = _dtanh(transfer) * d_activation
        d_embedding_space = datapoint.T @ d_transfer

        return d_embedding_space, d_weights

    @staticmethod
    def __update_parameter(d_parameter: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        return parameter - EMBEDDING_CFG["learn_rate"] * d_parameter

    def fit(self):
        for epoch in tqdm(range(EMBEDDING_CFG["epochs"])):
            epoch_loss = []

            for datapoint, label in zip(self.data, self.labels):
                transfer, activation, y_transfer, y_hat = self.__forward(datapoint)
                epoch_loss.append(self.__calculate_loss(y_hat, label))
                d_embedding, d_weights = self.__backprop(datapoint, label, transfer, activation, y_hat)
                self.embedding_space = self.__update_parameter(d_embedding, self.embedding_space)
                self.weights = self.__update_parameter(d_weights, self.weights)

            self.loss.append(sum(epoch_loss))

    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.show()

    def word_vector(self, word: str):
        return np.sum(word_matrix @ self.embedding_space, axis=1)
