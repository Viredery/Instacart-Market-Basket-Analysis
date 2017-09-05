import sys
import numpy as np
import pandas as pd

sys.path.append('../')
from param_config import config
from utils import pickle_load, pickle_dump

if __name__ == '__main__':
    print('Generating user_aisle features...')
    order_products_prior = pickle_load(config.order_products_prior_path)
    products = pickle_load(config.products_path)

    order_products_prior = pd.merge(order_products_prior, products, on='product_id', how='left')

    ua_feat = pd.DataFrame()
    ua_feat['ua_first_order'] = order_products_prior.groupby(["user_id", "aisle_id"])['order_number_before_last_order'].max()
    ua_feat['ua_last_order'] = order_products_prior.groupby(["user_id", "aisle_id"])['order_number_before_last_order'].min()
    ua_feat['ua_distinct_order_num'] = order_products_prior.groupby(['user_id', 'aisle_id']).order_id.nunique()
    ua_feat['ua_distinct_product_num'] = order_products_prior.groupby(['user_id', 'aisle_id'])['product_id'].nunique()

    feats = ['ua_first_order', 'ua_last_order', 'ua_distinct_order_num', 'ua_distinct_product_num']
    pickle_dump(ua_feat[feats], '{}/user_aisle_feat.pkl'.format(config.feat_folder))
    print('Done - user_aisle features')
