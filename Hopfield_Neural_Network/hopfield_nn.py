from copy import copy
from typing import List, Tuple, Union
import numpy as np


def signum_function(vector: np.ndarray) -> np.ndarray:
    """
    Apply signum on whole vector
    """
    return np.where(vector >= 0, 1, -1)


def make_network(neurons: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For fully connected graph of neurons, make their weights
    """
    W_triu = np.triu(np.random.choice([-1, 1], neurons), 1)
    W_tril = W_triu.T
    W = W_triu + W_tril

    return W


def update_weights(pattern: np.ndarray) -> np.ndarray:
    """
    Given a pattern vector returns weight matrix for memorized pattern
    """
    
    if pattern.shape[-1] == 1:
        return pattern * pattern.T - np.identity(pattern.shape[0])
    else:
        einstein_outer_prod = np.einsum('ab,ca->bca', pattern, pattern.T)   # einstein notation for matrix multiplicaions, ensures shapes you want
        outer_prod_sum = np.sum(einstein_outer_prod, axis=-1)
        np.fill_diagonal(outer_prod_sum, 0)
        return outer_prod_sum / pattern.shape[0]


def calculate_energy(neurons: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculates energy for given state of neurons
    """
    return -0.5*np.sum(weights * (neurons * neurons.T))


def learn_pattern(pattern: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Learns a given pattern and returns weights for the memorized pattern
    """

    weights = update_weights(pattern)

    return weights


def update_neuron_sync(neurons: np.ndarray, weights: np.ndarray, update_indexes: Union[List, np.ndarray]) -> np.ndarray:
    """
    Update state of neurons synchronously
    """
    if len(update_indexes) != 1:
        neurons[update_indexes] = signum_function( weights[update_indexes] @ neurons).reshape(-1, 1)
    else:
        neurons[update_indexes] = signum_function( weights[update_indexes] @ neurons)
    return neurons


def update_neuron_async(neurons: np.ndarray, weights: np.ndarray, update_indexes: Union[List, np.ndarray]) -> np.ndarray:
    """
    Update state of neurons asynchronously
    """
    for idx in update_indexes:
        neurons = update_neuron_sync(neurons, weights, [idx])

    return neurons


def recall_pattern(init_state: np.ndarray, weights: np.ndarray, firings: int, sync: bool = False) -> np.ndarray:
    """
    Given an initial state try to recall a pattern most similar to initial state
    """
    assert init_state.shape[0] == weights.shape[0], f"Initial state doesn't have the same number of neurons as weights do\nCurrently {init_state.shape[0]} needs to be {weights.shape[0]}"

    try:
        assert init_state.shape[1] == 1, "Initial state must be vector with dim[1] equal to 1"
    except IndexError:
        init_state = init_state.reshape(-1, 1)

    if sync:
        assert firings <= init_state.shape[0], f"For synchronous updating firings must be less than equal to number of neurons. Firings need to be {init_state.shape[0]} but are {firings}"

    indexes = np.arange(init_state.shape[0])
    neurons = init_state
    if sync:
        update_neuron_sync(neurons, weights, np.random.choice(indexes, firings))
    else:
        update_neuron_async(neurons, weights, np.random.choice(indexes, firings))

    energy = calculate_energy(neurons, weights)
    print(energy)
    return neurons


if __name__ == "__main__":
    pattern = np.array([
        [1.,-1.,-1.,-1.,-1.],
        [-1.,-1.,-1.,-1.,1.]
        ])
    test_state = np.array([1.,1.,-1.,-1.,-1.])
    w = make_network(5)
 
    w = learn_pattern(pattern, w)
    print()

    recalled = recall_pattern(test_state, w, 100, False)
    print()
    print(recalled)
