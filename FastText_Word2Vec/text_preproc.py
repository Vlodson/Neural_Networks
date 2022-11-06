import time
import zlib
import numpy as np
from typing import List, Dict, Text
import itertools
import re
import jsonpickle

from text import text, text2


def _one_hot_encode(index_dict: Dict[str, int]) -> Dict[str, np.ndarray]:
    """
    Uses the passed index_dict (ie gram_indexes) to create a new dictionairy with one hot encoded values instead of indexes
    """
    vectors = {}

    for gram in index_dict.keys():
        vector = np.zeros((1, max(index_dict.values())+1))
        vector[0, index_dict[gram]] = 1
        vectors[gram] = vector

    return vectors


def _gramify_word(word: str, grams: int) -> List[str]:
    """
    Takes a word and amount of letters for an ngram and returns a list of ngrams of the word plust the word itself
    """
    if len(word) <= grams:
        return [word]
    
    gram_list = [word[x:y] for x,y in itertools.combinations(range(len(word) + 1), r = 2) if len(word[x:y]) == grams] 
    gram_list.append(word)
    return gram_list


class Text_Preprocessing:
    """
    Class for all intermediate representation of a text corpus.
    Needed for training word embedding.
    """
    def __init__(self, corpus: str, forbidden: List[str]) -> None:
        """
        Parameters:
            corpus - text/corpus to be preprocessed
        
        Fields:
            - word_list - list of all words from corpus passed through a regex filter to remove clutter
            - word_indexes - dictionairy with all unique words from the corpus as keys and their index for one hot encoding as the value
            - gram_dict - dictionairy with unique words as keys and their n-grams as values
            - gram_index - dictionairy with all unique n-grams from the corpus as keys and their index for one hot encoding as the value

            - word_vectors - dictionairy with words as keys and their one hot encoded representation as values
            - gram_vectors - dictionairy with grams as keys and their one hot encoded representation as values

            - training_set - dictionairy with unique words as keys and for values a list with 2 elements
                             first element are all the one hot encoded n-grams of the word 
                             second element are all the one hot encoded context words for the word
            - gram_vocab_size - overall amount of unique ngrams
            - word_vocab_size - overall amount of unique words
        """
        self.corpus = corpus
        self.forbidden = forbidden
        
        self.word_list: List[str] = None
        self.word_indexes: Dict[str, int] = None
        self.gram_dict: Dict[str, List[str]] = None
        self.gram_indexes: Dict[str, int] = None

        self.word_vectors: Dict[str, np.ndarray] = None
        self.gram_vectors: Dict[str, np.ndarray] = None

        self.training_set: Dict[str, List[np.ndarray]] = None
        self.gram_vocab_size: int = None
        self.word_vocab_size: int = None


    def make_word_list(self):
        """
        Does regex cleaning of the corpus
        Removes punctuation
        Makes all words lowercase
        Removes all stop words
        """
        no_punct = re.sub("[.,?!/\\()*&^%$#@0-9]", "", self.corpus)
        to_lower = no_punct.lower()

        # technicly not needed but nice to have the whole preprocessing if the need arises
        # for the whole list of stop words: https://github.com/igorbrigadir/stopwords/blob/master/en/terrier.txt
        no_stops = re.sub(r"\band\b|\bis\b", "", to_lower)
        
        self.word_list = no_stops.split()


    def map_index_to_word(self):
        """
        Uses word_list to create the word_index field, finds unique words and gives them a unique index
        """
        self.word_indexes = {}
        index = 0

        for word in self.word_list:
            if word not in self.word_indexes.keys():
                self.word_indexes[word] = index
                index += 1


    def map_grams_to_word(self, grams: int):
        """
        Uses word_indexes to create gram_dict field, gramifys all unique words
        """
        self.gram_dict = {}
        
        for word in self.word_indexes.keys():
            self.gram_dict[word] = _gramify_word(word, grams)


    def map_index_to_gram(self):
        """
        Uses gram_dict to create gram_indexes, gives all unique grams a unique index
        """
        self.gram_indexes = {}
        index = 0

        for grams in self.gram_dict.values():
            for gram in grams:
                if gram not in self.gram_indexes.keys():
                    self.gram_indexes[gram] = index
                    index += 1


    def create_train_dataset(self, context_window: int, grams: int):
        """
        
        """
        self.make_word_list()
        self.map_index_to_word()
        self.map_grams_to_word(grams=grams)
        self.map_index_to_gram()
        
        self.gram_vectors = _one_hot_encode(self.gram_indexes)
        self.word_vectors = _one_hot_encode(self.word_indexes)

        self.gram_vocab_size = list(self.gram_vectors.values())[0].shape[1]
        self.word_vocab_size = list(self.word_vectors.values())[0].shape[1]
        self.training_set = {}

        for i, word in enumerate(self.word_list):
            
            if word in self.forbidden:
                continue
            
            word_grams = _gramify_word(word, grams)
            target_word = []
            context_words = []
            first_use = False

            if word not in self.training_set.keys():
                self.training_set[word] = [0,0]
                first_use = True

                for gram in word_grams:
                    target_word.append(self.gram_vectors[gram])

                target_word = np.array(target_word).reshape(-1, self.gram_vocab_size)
                self.training_set[word][0] = target_word

            lower = i-context_window
            if lower < 0:
                lower = 0

            for context_word in self.word_list[lower:i+context_window+1]:
                if context_word == word:
                    continue
                
                context_words.append(self.word_vectors[context_word])
            
            context_words = np.array(context_words).reshape(-1, self.word_vocab_size)
            if first_use:
                self.training_set[word][1] = context_words
            else:
                self.training_set[word][1] = np.append(self.training_set[word][1], context_words, axis=0)

    
    def serialize(self, path: str) -> None:
        json_str = jsonpickle.encode(self)
        json_comp = zlib.compress(json_str.encode())

        with open(path, "wb") as f:
            f.write(json_comp)


def _deserialize(path: str) -> Text_Preprocessing:
    with open(path, "rb") as f:
        json_str = zlib.decompress(f.read())

    text_preproc = jsonpickle.decode(json_str)

    return text_preproc


if __name__ == '__main__':
    zip_path = "text_preprocessing.zip"

    text_processor = Text_Preprocessing(text)
    
    start = time.time()
    text_processor.create_train_dataset(1, 3)
    print(time.time() - start)

    text_processor.serialize(zip_path)