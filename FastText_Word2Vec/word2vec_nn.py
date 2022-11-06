import time
import numpy as np
from typing import List, Tuple, Union
import matplotlib.pyplot as plt
import jsonpickle
import zlib

from text_preproc import Text_Preprocessing, _gramify_word
import text_preproc
from text import text, text2


def _softmax(tensor:np.ndarray):
    mxs = np.max(tensor, axis = 1).reshape(-1, 1)
    exps = np.exp(tensor - mxs)

    sums = np.sum(exps, axis = 1).reshape(-1, 1)

    return exps / sums


def _update(opti_param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray,
            omega1: float, omega2: float, eta: float, norm: float=1e-8) -> Tuple[np.ndarray]:
    new_m = omega1 * m + (1-omega1) * grad
    new_v = omega2 * v + (1-omega2) * (grad**2)

    mhat = new_m / (1-omega1)
    vhat = np.abs(new_v) / (1-omega2)

    opti_param -= (eta / (vhat + norm)**0.5) * mhat

    return opti_param, new_m, new_v


class word2vec:
    def __init__(self, space_dims: int, context_window: int, gram_size: int, forbidden: List[str], epochs: int, learn_rate: float, stop: float,
                text_preprocessing_path: Union[str, None] = None, verbose: bool=True):

        self.space_dims = space_dims
        self.context_window = context_window
        self.gram_size = gram_size
        self.forbidden = forbidden
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.stop = stop
        self.verbose = verbose

        # hold info for all things word related
        self.text_preprocessing_path = text_preprocessing_path
        if self.text_preprocessing_path is not None:
            self.text_processing: Text_Preprocessing = text_preproc._deserialize(self.text_preprocessing_path)
        else:
            self.text_processing: Text_Preprocessing = None

        # weights for the nn
        self.vector_space: np.ndarray = None
        self.w2: np.ndarray = None

        # network parameters
        self.h: np.ndarray = None
        self.yhat: np.ndarray = None
        self.dyz: np.ndarray = None
        self.dh: np.ndarray = None
        self.dvector_space: np.ndarray = None
        self.dw2: np.ndarray = None

        # losses
        self.L: List[float] = None
        self.epoch_L: List[float] = None
        self.context_L: float = None

        # adam params
        self.m_vector_space: np.ndarray = 0
        self.v_vector_space: np.ndarray = 0
        self.m_w2: np.ndarray = 0
        self.v_w2: np.ndarray = 0

    
    def create_training_set(self, corpus: str):
        self.text_processing = Text_Preprocessing(corpus, self.forbidden)
        self.text_processing.create_train_dataset(self.context_window, self.gram_size)


    def forward(self, target_word: np.ndarray):
        self.h = target_word @ self.vector_space
        yz = self.h @ self.w2
        self.yhat = _softmax(yz)

    
    # also adds epoch_L, to save on 1 for loop
    def back(self, target_word: np.ndarray, context_words: np.ndarray):
        self.context_L = 0
        self.dyz = 0
        for context in context_words:
            self.context_L += -np.sum(context*np.log(self.yhat)) / self.yhat.shape[0]
            self.dyz += self.yhat - context
        self.epoch_L.append(self.context_L)

        self.dw2 = self.h.T @ self.dyz / context_words.shape[0]
        self.dh = self.dyz @ self.w2.T
        self.dvector_space = target_word.T @ self.dh

    
    def train(self, corpus: str):
        # make training dataset
        if self.text_processing is None:
            self.create_training_set(corpus)
        
        # init weights
        self.vector_space = np.random.uniform(0, 1, (self.text_processing.gram_vocab_size, self.space_dims))
        self.w2 = np.random.uniform(0, 1, (self.space_dims, self.text_processing.word_vocab_size))

        # init losses
        self.L = []
        self.epoch_L = []

        for epoch in range(1, self.epochs+1):
            if (epoch % (self.epochs // 10) == 0) and self.verbose:
                print(f"Epoch {epoch} of {self.epochs}")

            # shuffle so the net doesnt learn the word pattern but the words themselves 
            words = list(self.text_processing.training_set.keys())
            np.random.shuffle(words)

            for word in words:
                target_word = self.text_processing.training_set[word][0]
                context_words = self.text_processing.training_set[word][1]

                self.forward(target_word)
                self.back(target_word, context_words)

                self.vector_space, self.m_vector_space, self.v_vector_space = \
                    _update(self.vector_space, self.dvector_space, self.m_vector_space, self.v_vector_space,
                    omega1=0.9, omega2=0.99, eta=self.learn_rate)

                self.w2, self.m_w2, self.v_w2 = _update(self.w2, self.dw2, self.m_w2, self.v_w2, omega1=0.9, omega2=0.99, eta=self.learn_rate)
            
            # early exit if loss diff low
            if epoch > 1:
                if (np.abs(self.L[-1] - np.sum(self.epoch_L)) <= self.stop):
                    break

            self.L.append(np.sum(self.epoch_L))
            self.epoch_L = []
        
        plt.plot(self.L)
        plt.show()


    def word_vector(self, word: str) -> np.ndarray:
        # gramifikuj
        # prodji kroz svaki vektor iz vector space i uzmi na tom indeksu vektor
        # ako nema tog grama onda samo preskoci
        # kad izvuces sve uradi prosek tih vektora
        word_gram = _gramify_word(word, self.gram_size)

        vector = np.zeros(self.space_dims)
        for gram in word_gram:
            try:
                vector += self.vector_space[self.text_processing.gram_indexes[gram]]
            except KeyError:
                continue
        
        if np.all(vector) == 0:
            print(f"No grams found for {word}")
            return None

        vector = np.reshape(vector, (1, -1))
        return np.mean(vector, axis=0)


    # similarity done by using cosine similarity
    def similar(self, word: str, top: int) -> List:
        word_vec = self.word_vector(word)

        if type(word_vec) is type(None):
            return None

        similarities = []
        for other_word in self.text_processing.word_indexes.keys():
            if other_word == word:
                continue
            
            other_vec = self.word_vector(other_word)
            
            numerator = word_vec @ other_vec
            denominator = np.linalg.norm(word_vec) * np.linalg.norm(other_vec)
            similarity = numerator / denominator

            similarities.append([similarity, other_word])

        similarities.sort(reverse=True)
        return similarities[:top]


    def serialize(self, path: str) -> None:
        # ? serialize text_preprocessing as well if first time loading one
        if self.text_preprocessing_path is None:
            new_path = path.split(".")[0] + "text_preproc." + path.split(".")[1] 
            self.text_processing.serialize(new_path)

            temp_txt_preproc = self.text_processing
            self.text_processing = None

            json_str = jsonpickle.encode(self)
            json_comp = zlib.compress(json_str.encode())

            with open(path, "wb") as f:
                f.write(json_comp)

            self.text_processing = temp_txt_preproc

        else:
            temp_txt_preproc = self.text_processing
            self.text_processing = None

            json_str = jsonpickle.encode(self)
            json_comp = zlib.compress(json_str.encode())

            with open(path, "wb") as f:
                f.write(json_comp)

            self.text_processing = temp_txt_preproc


def _deserialize(w2v_path: str, text_proc_path: str) -> word2vec:
    with open(w2v_path, "rb") as f:
        json_str = zlib.decompress(f.read())

    w2v_model: word2vec = jsonpickle.decode(json_str)
    text_processing = text_preproc._deserialize(text_proc_path)

    w2v_model.text_processing = text_processing

    return w2v_model


if __name__ == '__main__':    
    text_preproc_path = "text_preprocessing.zip"
    model_path = "model.zip"

    w2v = word2vec(space_dims=100, context_window=1, gram_size=3, epochs=int(3e1), learn_rate=1e-3, stop=1e-3, text_preprocessing_path=None, forbidden=[])
    
    start = time.time()
    w2v.train(text2)
    print(time.time() - start)

    w2v.serialize(model_path)

    t = _deserialize(model_path, text_preproc_path)

    print(w2v.similar("jova", 3))