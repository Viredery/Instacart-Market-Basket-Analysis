import numpy as np
import pandas as pd

from param_config import config
from utils import pickle_dump, pickle_load

'''user_product_feat = pd.read_csv('{}/user_product_feat.csv'.format(config.feat_folder))
prod_feat['prod_average_order_distance_per_rebuy'] = user_product_feat.groupby('product_id')['up_average_order_distance_per_rebuy'].mean()

user_product_feat_days = pd.read_csv('{}/user_product_feat_days.csv'.format(config.feat_folder))
prod_feat['prod_average_order_days_per_rebuy'] = user_product_feat_days.groupby('product_id')['up_average_order_days_per_rebuy'].mean()
'''



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
    product_feat['product_reorder_ratio'] = product_feat['product_first_reorder_num'] / product_feat['product_first_order_num']
    product_feat['prod_average_user_reorder_num'] = product_feat['product_reorder_num'] / product_feat['product_first_order_num']

    feats = ['product_order_num', 'product_reorder_num', 'product_reorder_frequency',
             'product_first_order_num', 'product_first_reorder_num', 'product_reorder_ratio',
             'prod_average_user_reorder_num']

    pickle_dump(product_feat[feats], '{}/product_feat.csv'.format(config.feat_folder))
    print('Done - product features')