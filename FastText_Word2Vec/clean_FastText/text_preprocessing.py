import string

import numpy as np

from typing import List, Dict, Tuple
from nltk.corpus import stopwords
from operator import itemgetter


def read_txt_file(file_path: str) -> str:
    with open(file_path, "r") as f:
        text = f.read()

    return text


def clean_punctuation(text: str) -> str:
    translator = str.maketrans(string.punctuation, '_' * len(string.punctuation))
    no_punctuation = text.translate(translator)
    return no_punctuation.replace('_', '')


def remove_whitespace(text: str) -> str:
    return " ".join(text.split())


def remove_digits(text: str) -> str:
    return ''.join(c for c in text if not c.isdigit())


def lower_case_all(text: str) -> str:
    return text.lower()


def tokenize(text: str) -> List[str]:
    """
    Returns tokenized list sorted alphabetically
    """
    tokens = text.split(' ')
    tokens.sort()
    return tokens


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [token for token in tokens if token not in stopwords.words()]


def remove_short_tokens(tokens: List[str], ngrams: int) -> List[str]:
    return [token for token in tokens if len(token) >= ngrams]


def clean_corpus(corpus_path: str, ngrams: int) -> List[str]:
    """
    Reads a txt file;
    Cleans it from punctuation;
    Removes extra whitespace;
    Turns all words to lowercase;
    Tokenizes the words;
    Removes stopwords from token list
    Returns a cleaned tokenized corpus
    :param corpus_path:
    :param ngrams:
    :return tokens:
    """
    corpus = read_txt_file(corpus_path)
    no_punctuation = clean_punctuation(corpus)
    no_whitespace = remove_whitespace(no_punctuation)
    no_digits = remove_digits(no_whitespace)
    lower_case = lower_case_all(no_digits)

    tokens = tokenize(lower_case)
    tokens = remove_stopwords(tokens)
    tokens = remove_short_tokens(tokens, ngrams)
    return tokens


def word_ngram(word: str, ngrams: int) -> List[str]:
    """
    Returns a list of ngrams from word as well as the word itself
    :param word:
    :param ngrams:
    :return:
    """
    gram_list = [word]
    grams = [word[i:i + ngrams] for i in range(len(word) - ngrams + 1)]
    gram_list.extend(grams)
    return gram_list


def create_ngram_list(tokens: List[str], ngrams: int) -> List[str]:
    ngram_list = []
    for token in tokens:
        ngram_list.extend(word_ngram(token, ngrams))

    return ngram_list


def one_hot_encode(ngram_list: List[str]) -> Dict[str, np.ndarray]:
    unique_ngram_list = np.unique(ngram_list)
    one_hot_matrix = np.eye(len(unique_ngram_list))
    ngram_vector_dict = {}
    for i, ngram in enumerate(unique_ngram_list):
        ngram_vector_dict[ngram] = one_hot_matrix[i]

    return ngram_vector_dict


def get_context(tokens: List[str], token_idx: int, ngram_dict: Dict[str, np.ndarray],
                context_window: int) -> np.ndarray:
    lower_ctx = token_idx - context_window if token_idx - context_window >= 0 else 0
    upper_ctx = token_idx + context_window + 1 if token_idx + context_window + 1 <= len(tokens) else len(tokens)
    context = tokens[lower_ctx:upper_ctx]
    _ = context.pop(len(context) // 2) if lower_ctx != 0 else context.pop(0)
    context = np.sum(itemgetter(*context)(ngram_dict), axis=0).reshape(1, -1)

    return context


def create_dataset(tokens: List[str], ngram_dict: Dict[str, np.ndarray], context_window: int, ngrams: int) -> \
        Dict[str, List[np.ndarray]]:
    dataset = {}

    for idx, token in enumerate(tokens):
        datapoint = np.concatenate([itemgetter(*word_ngram(token, ngrams))(ngram_dict)], axis=0)
        label = get_context(tokens, idx, ngram_dict, context_window)
        dataset[token] = [datapoint, label]

    return dataset


def preprocess_text(corpus_path: str, ngrams: int, context_window: int) -> \
        Tuple[Dict[str, List[np.ndarray]], Dict[str, np.ndarray]]:
    token_list = clean_corpus(corpus_path, ngrams)
    ngrams_list = create_ngram_list(token_list, ngrams)
    ngram_vectors = one_hot_encode(ngrams_list)
    dataset = create_dataset(token_list, ngram_vectors, context_window=context_window, ngrams=ngrams)

    return dataset, ngram_vectors
