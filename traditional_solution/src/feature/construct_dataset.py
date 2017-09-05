import sys
import numpy as np
import pandas as pd

sys.path.append('../')
from param_config import config
from utils import pickle_load, pickle_dump

if __name__ == '__main__':
    print('Construct train/test dataset...')
    order_products_prior = pickle_load(config.order_products_prior_path)
    order_products_train = pickle_load(config.order_products_train_path)
    orders = pickle_load(config.orders_path)

    train_orders = orders[orders.eval_set == 'train'][['order_id', 'user_id']].copy()
    test_orders = orders[orders.eval_set == 'test'][['order_id', 'user_id']].copy()

    user_product_pair = order_products_prior[['user_id', 'product_id']].drop_duplicates()

    train_df = pd.merge(train_orders, user_product_pair, on='user_id')
    test_df = pd.merge(test_orders, user_product_pair, on='user_id')
    
    order_products_train = order_products_train[['order_id', 'product_id', 'reordered']]
    train_df = pd.merge(train_df, order_products_train, on=['order_id', 'product_id'], how='left')
    train_df['reordered'] = train_df['reordered'].fillna(0).astype(np.int)
    
    x_train = train_df[['order_id', 'user_id', 'product_id']]
    y_train = train_df['reordered']
    
    x_test = test_df
    
    pickle_dump(x_train, '{}/x_train.pkl'.format(config.output_folder))
    pickle_dump(y_train, '{}/y_train.pkl'.format(config.output_folder))
    pickle_dump(x_test, '{}/x_test.pkl'.format(config.output_folder))
    print('Done - dataset construction')