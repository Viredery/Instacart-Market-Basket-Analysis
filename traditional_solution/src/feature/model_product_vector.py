import sys
import numpy as np
import pandas as pd

sys.path.append('../')
from param_config import config
from utils import Embedding, pickle_load, pickle_dump

def gen_product_sentences():

    order_products_prior = pickle_load(config.order_products_prior_path)[['order_id', 'product_id', 'add_to_cart_order']].copy()
    order_products_train = pickle_load(config.order_products_train_path)[['order_id', 'product_id', 'add_to_cart_order']].copy()

    order_products_train["product_id"] = order_products_train["product_id"].astype(str)
    order_products_prior["product_id"] = order_products_prior["product_id"].astype(str)

    product_sentences_train = order_products_train.sort_values(['order_id', 'add_to_cart_order']).groupby(
            'order_id').apply(lambda order: order['product_id'].tolist())
    product_sentences_prior = order_products_prior.sort_values(['order_id', 'add_to_cart_order']).groupby(
            'order_id').apply(lambda order: order['product_id'].tolist())

    product_sentences = product_sentences_prior.append(product_sentences_train).values

    return product_sentences

if __name__ == '__main__':
    print('Generating sentences...')
    product_sentences = gen_product_sentences()

    print('Generating product_vector features...')
    embedding = Embedding(product_sentences)
    embedding.word_to_vector(size=100, window=5, min_count=2)
    embedding.reduce_dimension(n_components=2)
    product_vector_feat = embedding.return_dataframe(name='product_id')

    product_vector_feat['product_id'] = product_vector_feat['product_id'].astype(np.int)
    product_vector_feat['product_id_vector_1'] = product_vector_feat['product_id_vector_1'].astype(np.float)
    product_vector_feat['product_id_vector_2'] = product_vector_feat['product_id_vector_1'].astype(np.float)
    product_vector_feat.set_index('product_id', inplace=True)

    pickle_dump(product_vector_feat, '{}/product_vector_feat.pkl'.format(config.feat_folder))
    print('Done - product_vector features')