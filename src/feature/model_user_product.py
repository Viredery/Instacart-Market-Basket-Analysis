import numpy as np
import pandas as pd

from param_config import config
from utils import pickle_load, pickle_dump


if __name__ == '__main__':
    print('Generating user_product features...')
    order_products_prior = pickle_load(config.order_products_prior_path)

    up_feat = pd.DataFrame()
    up_feat['up_order_num'] = order_products_prior.groupby(['user_id', 'product_id']).size()
    up_feat['up_average_add_to_cart_order'] = order_products_prior.groupby(['user_id', 'product_id']).add_to_cart_order.mean()

    # features based on order_number_before_last_order 
    up_feat['up_first_order'] = order_products_prior.groupby(['user_id', 'product_id']).order_number_before_last_order.max()
    up_feat['up_last_order'] = order_products_prior.groupby(['user_id', 'product_id']).order_number_before_last_order.min()
    up_feat['up_average_order'] = order_products_prior.groupby(['user_id', 'product_id']).order_number_before_last_order.mean()
    up_feat['up_average_order_distance'] = (up_feat['up_first_order'] - up_feat['up_last_order']) / (up_feat['up_order_num'] - 1)
    up_feat['up_order_skew'] = up_feat['up_last_order'] / up_feat['up_average_order']

    # features based on order_days_before_last_order
    up_feat['up_first_order_days'] = order_products_prior.groupby(['user_id', 'product_id']).order_days_before_last_order.max()
    up_feat['up_last_order_days'] = order_products_prior.groupby(['user_id', 'product_id']).order_days_before_last_order.min()
    up_feat['up_average_order_days'] = order_products_prior.groupby(['user_id', 'product_id']).order_days_before_last_order.mean()
    up_feat['up_average_order_days_distance'] = (up_feat['up_first_order_days'] - up_feat['up_last_order_days']) / (up_feat['up_order_num'] - 1)
    up_feat['up_order_days_skew'] =  up_feat['up_last_order_days'] / up_feat['up_average_order_days']

    up_feat['up_last_order_days_ratio'] = up_feat['up_last_order_days'] / up_feat['up_average_order_days_distance']
    up_feat['up_last_order_days_diff'] = up_feat['up_last_order_days'] - up_feat['up_average_order_days_distance']

    # Fillna
    up_feat['up_average_order_distance'].fillna(999, inplace=True)
    up_feat['up_average_order_days_distance'].fillna(999, inplace=True)
    # these fetures still have null values:
    #     up_order_days_skew, up_last_order_days_ratio, up_last_order_days_diff


    feats = ['up_order_num', 'up_average_add_to_cart_order', 'up_first_order', 'up_last_order', 'up_average_order',
             'up_average_order_distance', 'up_order_skew', 'up_first_order_days', 'up_last_order_days',
             'up_average_order_days', 'up_average_order_days_distance', 'up_order_days_skew', 'up_last_order_days_ratio',
             'up_last_order_days_diff']

    pickle_dump(up_feat[feats], '{}/user_product_feat.pkl'.format(config.feat_folder))
    print('Done - user_product features')
