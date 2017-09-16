import copy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataFrame(object):
    """Minimal pd.DataFrame
    
    support for shuffling, batching, and train/test splitting

    parameters
    ----------
    columns : list
        List of names with the same length as the data.
    data : list
        List of n-dimensional data matrices. All matrices must have the same leading dimension.
 
    """


    def __init__(self, columns, data):
        assert len(columns) == len(data)

        lengths = list(set([mat.shape[0] for mat in data]))
        assert len(lengths) == 1

        self.columns = columns
        self.data = data
        self.length = lengths[0]
        self.dict = dict(zip(self.columns, self.data))
        self.idx = np.arange(self.length)

    def dtypes(self):
        return pd.Series(dict(zip(copy.copy(self.columns), [mat.dtypes for mat in self.data])))

    def shape(shape):
        return pd.Series(dict(zip(copy.copy(self.columns), [mat.shape for mat in self.data])))

    def mask(self, mask):
        return DataFrame(copy.copy(self.columns), [mat[mask] for mat in self.data])

    def shuffle(self):
        np.random.shuffle(self.idx)


    def batch_generator(self, batch_size, shuffle=True, num_epochs=10000, allow_smaller_final_batch=False):
        epoch_num = 0
        while epoch_num < num_epochs:
            if shuffle:
                self.shuffle()

            for i in range(0, self.length, batch_size):
                batch_idx = self.idx[i : i + batch_size]
                if not allow_smaller_final_batch and len(batch_idx) != batch_size:
                    break
                yield DataFrame(copy.copy(self.columns), [mat[batch_idx] for mat in self.data])

            epoch_num += 1

    def train_test_split(self, train_size, random_state=np.random.randint(10000)):
        train_idx, test_idx = train_test_split(self.idx, train_size=train_size, random_state=random_state)
        train_df = DataFrame(copy.copy(self.columns), [mat[train_idx] for mat in self.data])
        test_df = DataFrame(copy.copy(self.columns), [mat[test_idx] for mat in self.data])
        return train_df, test_df

    def iterrows(self):
        for i in self.idx:
            yield self[i]

    def __iter__(self):
        return self.dict.items().__iter__()

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.dict[key]
        elif isinstance(key, int):
            return pd.Series(dict(zip(self.columns, [mat[self.idx[key]] for mat in self.data])))

    def __setitem__(self, key, value):
        assert value.shape[0] == len(self), 'matrix first dimension does not match'
        if key not in self.columns:
            self.columns.append(key)
            self.data.append(value)
        self.dict[key] = value
