import numpy as np
import pandas as pd

from param_config import config
from utils import pickle_load, pickle_dump

if __name__ == '__main__':
    print('Generating user_product features based on other features...')

    up_feat = pickle_load('{}/user_product_feat.pkl'.format(config.feat_folder))
    user_feat = pickle_load('{}/user_feat.pkl'.format(config.feat_folder))
    up_feat = pd.merge(up_feat.reset_index(), user_feat.reset_index(),
            on='user_id', how='left').set_index(['user_id', 'product_id'], drop=True)

    up_feat['up_order_num_ratio'] = up_feat['up_order_num'] / up_feat['user_order_num']
    up_feat['up_order_num_proportion'] = up_feat['up_order_num'] / up_feat['user_total_product_num']
    up_feat['up_average_add_to_cart_order_ratio'] = up_feat['up_average_add_to_cart_order'] / up_feat['user_average_product_num']

    # features based on order_number_before_last_order and other features
    up_feat['up_first_order_proportion'] = up_feat['up_first_order'] / up_feat['user_order_num']
    up_feat['up_last_order_proportion'] = up_feat['up_last_order'] / up_feat['user_order_num']
    up_feat['up_average_order_proportion'] = up_feat['up_average_order'] / up_feat['user_order_num']
    up_feat['up_last_order_proportion_ratio'] = up_feat['up_last_order_proportion'] / up_feat['product_average_order_distance']

    # features based on order_days_before_last_order and other features
    up_feat['up_first_order_days_proportion'] = up_feat['up_first_order_days'] / up_feat['user_order_days']
    up_feat['up_last_order_days_proportion'] = up_feat['up_last_order_days'] / up_feat['user_order_days']
    up_feat['up_average_order_days_proportion'] = up_feat['up_average_order_days'] / up_feat['user_order_days']
    up_feat['up_last_order_days_proportion_ratio'] = \
            up_feat['up_last_order_days_proportion'] / up_feat['up_average_order_days_distance']

    feats = ['up_order_num_ratio', 'up_order_num_proportion', 'up_average_add_to_cart_order_ratio',
             'up_first_order_proportion', 'up_last_order_proportion','up_average_order_proportion',
             'up_last_order_proportion_ratio', 'up_first_order_days_proportion', 'up_last_order_days_proportion',
             'up_average_order_days_proportion', 'up_last_order_days_proportion_ratio']
    pickle_dump(up_feat[feats], '{}/user_product_dependent_feat.pkl'.format(config.feat_folder))
    print('Done - user_product features based on other features')    