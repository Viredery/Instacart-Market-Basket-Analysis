import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.decomposition import PCA

class Embedding(object):

    def __init__(self, sentences=None):
        self.sentences = sentences
        self.word_embedding = None

    def word_to_vector(self, size=100, window=5, min_count=2, save_path=None):
        model = Word2Vec(self.sentences, size=size, window=window, min_count=min_count, workers=4)
        if save_path != None:
            model.save(save_path)
        self.word_embedding = np.hstack((np.array(model.wv.index2word).reshape(-1, 1),
                                         model.wv.syn0))

    def load_model(self, path):
        model = Word2Vec.load(path)
        self.word_embedding = np.hstack((np.array(model.wv.index2word).reshape(-1, 1),
                                         model.wv.syn0))

    def reduce_dimension(self, n_components):
        if self.word_embedding is None:
            return
        pca = PCA(n_components=n_components)
        self.word_embedding = np.hstack((self.word_embedding[:, 0].reshape(-1,1), 
                                         pca.fit_transform(self.word_embedding[:, 1:])))

    def return_dataframe(self, name='word'):
        if self.word_embedding is None:
            return
        col_len = self.word_embedding.shape[1]
        col_names = [name] + ['{}_vector_{}'.format(name, i) for i in range(1, col_len)]
        return pd.DataFrame(self.word_embedding, columns=col_names)


class MacOSFile(object):

    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        buffer = bytearray(n)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31-1)
            buffer[idx : idx+batch_size] = self.f.read(batch_size)
            idx += batch_size
        return buffer

    def write(self, buffer):
        n = len(buffer)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31-1)
            self.f.write(buffer[idx : idx+batch_size])
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))
