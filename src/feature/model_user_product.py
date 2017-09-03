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

    up_feat['up_last_order_ratio'] = up_feat['up_last_order'] / up_feat['up_average_order_distance']
    up_feat['up_last_order_diff'] = up_feat['up_last_order'] - up_feat['up_average_order_distance']

    up_feat['product_average_order_distance'] = up_feat.reset_index().groupby(
            'product_id')['up_average_order_distance'].transform('mean').values
    up_feat['up_last_order_ratio_based_on_product'] = up_feat['up_last_order'] / up_feat['product_average_order_distance']

    # features based on order_days_before_last_order
    up_feat['up_first_order_days'] = order_products_prior.groupby(['user_id', 'product_id']).order_days_before_last_order.max()
    up_feat['up_last_order_days'] = order_products_prior.groupby(['user_id', 'product_id']).order_days_before_last_order.min()
    up_feat['up_average_order_days'] = order_products_prior.groupby(['user_id', 'product_id']).order_days_before_last_order.mean()
    up_feat['up_average_order_days_distance'] = (up_feat['up_first_order_days'] - up_feat['up_last_order_days']) \
            / (up_feat['up_order_num'] - 1)
    up_feat['up_order_days_skew'] =  up_feat['up_last_order_days'] / up_feat['up_average_order_days']

    up_feat['up_last_order_days_ratio'] = up_feat['up_last_order_days'] / up_feat['up_average_order_days_distance']
    up_feat['up_last_order_days_diff'] = up_feat['up_last_order_days'] - up_feat['up_average_order_days_distance']

    up_feat['product_average_order_days_distance'] = up_feat.reset_index().groupby(
            'product_id')['up_average_order_days_distance'].transform('mean').values
    up_feat['up_last_order_days_ratio_based_on_product'] = up_feat['up_last_order_days'] / up_feat['product_average_order_days_distance']

    # Fillna
    up_feat['up_average_order_distance'].fillna(999, inplace=True)
    up_feat['up_average_order_days_distance'].fillna(999, inplace=True)
    up_feat['up_order_days_skew'].fillna(1, inplace=True)

    # up_hour_of_day_distance, up_dow_distance
    def get_up_last_order_id(group):
        return group[group.order_number_before_last_order == group.order_number_before_last_order.min()]['order_id'].values[0]
    up_feat['order_id'] = order_products_prior.groupby(['user_id', 'product_id']).apply(get_up_last_order_id)
    up_feat = up_feat.reset_index()

    orders = pickle_load(config.orders_path)[['order_id', 'user_id', 'eval_set', 'order_dow', 'order_hour_of_day']]
    up_feat = pd.merge(up_feat, orders[['user_id', 'order_id', 'order_dow', 'order_hour_of_day']],
            on=['user_id', 'order_id'], how='left')
    up_feat.rename(columns={'order_dow': 'up_last_order_dow', 'order_hour_of_day': 'up_last_order_hour_of_day'}, inplace=True)
    
    predicted_orders = orders[orders.eval_set != 'prior']
    up_feat = pd.merge(up_feat, predicted_orders[['user_id', 'order_dow', 'order_hour_of_day']],
            on='user_id', how='left')
    
    orders.set_index('order_id', inplace=True)
    
    up_feat['up_hour_of_day_distance'] = np.abs(up_feat.order_hour_of_day - up_feat.order_id.map(orders.order_hour_of_day)).map(
            lambda x: min(x, 24-x))
    up_feat['up_dow_distance'] = np.abs(up_feat.order_dow - up_feat.order_id.map(orders.order_dow)).map(
            lambda x: min(x, 7-x))
    
    up_feat.drop(['order_id', 'order_dow', 'order_hour_of_day', 'up_last_order_dow', 'up_last_order_hour_of_day'],
            axis=1, inplace=True)
    up_feat.set_index(['user_id', 'product_id'], inplace=True)


    feats = ['up_order_num', 'up_average_add_to_cart_order', 'up_first_order', 'up_last_order', 'up_average_order',
             'up_average_order_distance', 'up_order_skew', 'up_last_order_ratio', 'up_last_order_diff',
             'product_average_order_distance', 'up_last_order_ratio_based_on_product', 
             'up_first_order_days', 'up_last_order_days', 'up_average_order_days', 'up_average_order_days_distance',
             'up_order_days_skew', 'up_last_order_days_ratio', 'up_last_order_days_diff',
             'product_average_order_days_distance', 'up_last_order_days_ratio_based_on_product',
             'up_hour_of_day_distance', 'up_dow_distance']

    pickle_dump(up_feat[feats], '{}/user_product_feat.pkl'.format(config.feat_folder))
    print('Done - user_product features')
