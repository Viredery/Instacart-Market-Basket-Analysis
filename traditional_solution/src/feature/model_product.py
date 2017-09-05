import sys
import numpy as np
import pandas as pd

sys.path.append('../')
from param_config import config
from utils import pickle_dump, pickle_load

if __name__ == '__main__':
    print('Generating product features...')
    order_products_prior = pickle_load(config.order_products_prior_path)

    order_products_prior['user_product_order_number'] = order_products_prior.sort_values(
            by=['user_id', 'product_id', 'order_number']).groupby(['user_id', 'product_id']).cumcount()

    product_feat = pd.DataFrame()
    product_feat['product_order_num'] = order_products_prior.groupby('product_id').size()
    product_feat['product_reorder_num'] = order_products_prior.groupby('product_id')['reordered'].sum()
    product_feat['product_reorder_frequency'] = product_feat['product_reorder_num'] / product_feat['product_order_num']

    product_feat['product_first_order_num'] = order_products_prior[order_products_prior.user_product_order_number == 0].groupby(
            'product_id').size()
    product_feat['product_first_reorder_num'] = order_products_prior[order_products_prior.user_product_order_number == 1].groupby(
            'product_id').size()
    product_feat['product_first_reorder_num'].fillna(0, inplace=True) # fillna

    product_feat['product_user_order_only_once_num'] = \
            product_feat['product_first_order_num'] - product_feat['product_first_reorder_num']
    product_feat['product_user_order_only_once_ratio'] = \
            product_feat['product_user_order_only_once_num'] / product_feat['product_first_order_num']

    product_feat['product_reorder_ratio'] = product_feat['product_first_reorder_num'] / product_feat['product_first_order_num']
    product_feat['product_average_user_reorder_num'] = product_feat['product_reorder_num'] / product_feat['product_first_order_num']
    product_feat['product_average_add_to_cart_order'] = order_products_prior.groupby('product_id')['add_to_cart_order'].mean()

    feats = ['product_order_num', 'product_reorder_num', 'product_reorder_frequency',
             'product_first_order_num', 'product_first_reorder_num', 'product_reorder_ratio',
             'product_user_order_only_once_num', 'product_user_order_only_once_ratio',
             'product_average_user_reorder_num', 'product_average_add_to_cart_order']

    pickle_dump(product_feat[feats], '{}/product_feat.pkl'.format(config.feat_folder))
    print('Done - product features')