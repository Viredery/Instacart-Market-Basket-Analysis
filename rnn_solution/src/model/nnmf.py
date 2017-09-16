import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
import sys
sys.path.append('../')
from param_config import config
from data_frame import DataFrame


class DataReader(object):
    
    def __init__(self, data):
        columns = ['i', 'j', 'V_ij']
        df = DataFrame(columns=columns, data=data)
        self.train_df, self.val_df = df.train_test_split(train_size=0.9, random_state=config.random_seed)
        self.test_df = df

        self.num_users = df['i'].max() + 1
        self.num_products = df['j'].max() + 1

    def train_batch_generator(self, batch_size):
        return self.batch_generator(df=self.train_df, batch_size=batch_size, shuffle=True, num_epochs=10000)

    def val_batch_generator(self, batch_size):
        return self.batch_generator(df=self.val_df, batch_size=batch_size, shuffle=True, num_epochs=10000)

    def test_batch_generator(self, batch_size):
        return self.batch_generator(df=self.test_df, batch_size=batch_size, shuffle=True, num_epochs=10000)

    def batch_generator(self, df, batch_size, shuffle, num_epochs, is_test=False):
        return df.batch_generator(batch_size=batch_size, shuffle=shuffle, num_epochs=num_epochs, allow_smaller_final_batch=is_test)




class NNMF(object):
    """Non-Negative Matrix Factorization (NNMF)
    """

    def __init__(self, 
                 data_reader,
                 rank, 
                 learning_rate=0.005, 
                 batch_size=4096,
                 num_training_steps=150000, 
                 early_stopping_steps=30000,
                 log_interval=200):

        self.reader = data_reader
        self.rank = rank
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_training_steps = num_training_steps
        self.early_stopping_steps = early_stopping_steps
        self.log_interval = log_interval


    def fit(self):
        self.i = tf.placeholder(dtype=tf.int32, shape=[None])
        self.j = tf.placeholder(dtype=tf.int32, shape=[None])
        self.V_ij = tf.placeholder(dtype=tf.float32, shape=[None])

        self.W = tf.Variable(tf.truncated_normal([self.reader.num_users, self.rank]))
        self.H = tf.Variable(tf.truncated_normal([self.reader.num_products, self.rank]))
        W_bias = tf.Variable(tf.truncated_normal([self.reader.num_users]))
        H_bias = tf.Variable(tf.truncated_normal([self.reader.num_products]))

        global_mean = tf.Variable(0.0)

        w_i = tf.gather(self.W, self.i)
        h_j = tf.gather(self.H, self.j)

        w_bias = tf.gather(W_bias, self.i)
        h_bias = tf.gather(H_bias, self.j)

        interaction = tf.reduce_sum(w_i * h_j, reduction_indices=1)
        preds = global_mean + w_bias + h_bias + interaction


        rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(preds, self.V_ij)))
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(rmse)

        clip_W = self.W.assign(tf.maximum(tf.zeros_like(self.W), self.W))
        clip_H = self.H.assign(tf.maximum(tf.zeros_like(self.H), self.H))
        clip = tf.group(clip_W, clip_H)

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
                
                _, val_loss = self.session.run(fetches=[clip, rmse], feed_dict=val_feed_dict)
                #[val_loss] = self.session.run(fetches=[rmse], feed_dict=val_feed_dict)
                val_loss_history.append(val_loss)
                
                # train step
                train_batch_df = next(train_generator)
                train_feed_dict = { getattr(self, placeholder_name, None): data
                                    for placeholder_name, data in train_batch_df if hasattr(self, placeholder_name)}

                self.session.run(fetches=[train_step], feed_dict=train_feed_dict)
                _, train_loss = self.session.run(fetches=[clip, rmse], feed_dict=train_feed_dict)

                #train_loss, _ = self.session.run(fetches=[rmse, train_step], feed_dict=train_feed_dict)
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
        np.save('{}/user_embeddings.npy'.format(config.data_store_path), self.W.eval(self.session))
        np.save('{}/product_embeddings.npy'.format(config.data_store_path), self.H.eval(self.session))


if __name__ == '__main__':

    prior_products = pd.read_csv(config.raw_order_products_prior_path, usecols=['order_id', 'product_id'])
    orders = pd.read_csv(config.raw_orders_path, usecols=['user_id', 'order_id'])

    user_products = prior_products.merge(orders, how='left', on='order_id')

    counts = user_products.groupby(['user_id', 'product_id']).size().rename('count').reset_index()

    i = counts['user_id'].values.copy()
    j = counts['product_id'].values.copy()
    V_ij = counts['count'].values.copy()

    dr = DataReader(data=[i, j, V_ij])
    nnmf = NNMF(data_reader=dr, rank=25)

    del prior_products, orders, user_products, counts, i, j, V_ij

    nnmf.fit()
    nnmf.predict()

