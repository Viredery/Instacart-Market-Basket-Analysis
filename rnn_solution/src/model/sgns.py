import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import math
from collections import deque

sys.path.append('../')
from param_config import config
from data_frame import DataFrame


class DataReader(object):
    
    def __init__(self, data):
        columns = ['x', 'y']
        df = DataFrame(columns=columns, data=data)
        self.train_df, self.val_df = df.train_test_split(train_size=0.9, random_state=config.random_seed)

        self.num_products = df['y'].max() + 1
        self.product_dist = np.bincount(df['y']).tolist()

    def train_batch_generator(self, batch_size):
        return self.batch_generator(df=self.train_df, batch_size=batch_size, shuffle=True, num_epochs=10000)

    def val_batch_generator(self, batch_size):
        return self.batch_generator(df=self.val_df, batch_size=batch_size, shuffle=True, num_epochs=10000)

    def batch_generator(self, df, batch_size, shuffle, num_epochs, is_test=False):
        return df.batch_generator(batch_size=batch_size, shuffle=shuffle, num_epochs=num_epochs, allow_smaller_final_batch=is_test)


class SGNS(object):
    """Skip-Gram with Negative Sampling
    """

    def __init__(self, 
                 data_reader,
                 embedding_size,
                 num_sampled,
                 learning_rate=0.002, 
                 batch_size=64,
                 num_training_steps=10*10**6, 
                 early_stopping_steps=100000,
                 log_interval=500):

        self.reader = data_reader
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_training_steps = num_training_steps
        self.early_stopping_steps = early_stopping_steps
        self.log_interval = log_interval

        self.vocabulary_size = self.reader.num_products


    def fit(self):
        self.x = tf.placeholder(tf.int32, [None])
        self.y = tf.placeholder(tf.int32, [None])

        self.embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        nce_weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, self.embedding_size], stddev=1.0 / math.sqrt(self.embedding_size)))
        nce_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

        embed = tf.nn.embedding_lookup(self.embeddings, self.x)

        sampled_values = tf.nn.fixed_unigram_candidate_sampler(
            true_classes=tf.cast(tf.reshape(self.y, (-1, 1)), tf.int64),
            num_true=1,
            num_sampled=self.num_sampled,
            unique=True,
            range_max=self.vocabulary_size,
            distortion=0.75,
            unigrams=self.reader.product_dist
        )

        loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=nce_weights,
                               biases=nce_biases,
                               labels=self.y,
                               inputs=embed,
                               num_sampled=self.num_sampled,
                               num_classes=self.vocabulary_size,
                               sampled_values=sampled_values))

        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)


        self.session = tf.Session()
        with self.session.as_default():
            self.session.run(tf.global_variables_initializer())

            train_generator = self.reader.train_batch_generator(self.batch_size)
            val_generator = self.reader.val_batch_generator(self.batch_size)

            train_loss_history = deque(maxlen=self.log_interval)
            val_loss_history = deque(maxlen=self.log_interval)
            best_validation_loss, best_validation_tstep = float('inf'), 0

            step = 0
            while step < self.num_training_steps:

                # validation evaluation
                val_batch_df = next(val_generator)

                val_feed_dict = { getattr(self, placeholder_name, None): data
                                    for placeholder_name, data in val_batch_df if hasattr(self, placeholder_name)}
                
                [val_loss] = self.session.run(fetches=[loss], feed_dict=val_feed_dict)
                #print(val_loss)
                val_loss_history.append(val_loss)
            
                # train step
                train_batch_df = next(train_generator)
                train_feed_dict = { getattr(self, placeholder_name, None): data
                                    for placeholder_name, data in train_batch_df if hasattr(self, placeholder_name)}

                train_loss, _ = self.session.run(fetches=[loss, train_step], feed_dict=train_feed_dict)
                train_loss_history.append(train_loss)

                if step % self.log_interval == 0:
                    avg_train_loss = sum(train_loss_history) / len(train_loss_history)
                    avg_val_loss = sum(val_loss_history) / len(val_loss_history)
                    print('[[step {}]] [train] {} [val] {}'.format(step, avg_train_loss, avg_val_loss))
                    if avg_val_loss < best_validation_loss:
                        best_validation_loss = avg_val_loss
                        best_validation_tstep = step

                    if step - best_validation_tstep > self.early_stopping_steps:
                        return

                step += 1

    def predict(self):
        np.save('{}/word2vec_product_embeddings.npy'.format(config.data_store_path), self.embeddings.eval(self.session))

if __name__ == '__main__':

    user_data = pd.read_csv(config.users_path)

    x = []
    y = []
    for _, row in user_data.iterrows():
        if _ % 10000 == 0:
            print(_)

        user_id = row['user_id']
        products = row['product_ids']
        products = ' '.join(products.split()[:-1])
        for order in products.split():
            items = order.split('_')
            for i in range(len(items)):
                for j in range(max(0, i - 2), min(i + 3, len(items))): # window = 5
                    if i != j:
                        x.append(int(items[i])) # input
                        y.append(int(items[j])) # label (context)


    x = np.array(x)
    y = np.array(y)


    dr = DataReader(data=[x, y])
    sgns = SGNS(data_reader=dr, embedding_size=25, num_sampled=100)

    del user_data, x, y

    sgns.fit()
    sgns.predict()