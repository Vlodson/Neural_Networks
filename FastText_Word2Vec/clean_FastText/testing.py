from text_preprocessing import preprocess_text
from word_embedding import WordEmbedding
from global_cfg import EMBEDDING_CFG

path = r"C:\Users\vlada\Desktop\ML_DS\Algorithms\Neural_Networks\FastText_Word2Vec\clean_FastText\test_text.txt"
dataset, one_hots = preprocess_text(path, EMBEDDING_CFG["ngrams"], EMBEDDING_CFG["context_window"])
m = WordEmbedding(dataset, one_hots)
m.fit()
m.plot_loss()
print(m.word_vector("industry").shape)
