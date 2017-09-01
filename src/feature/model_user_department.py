import pandas as pd
import numpy as np

from param_config import config
from utils import pickle_load, pickle_dump

if __name__ == '__main__':
    print('Generating user_department features...')
    order_products_prior = pickle_load(config.order_products_prior_path)
    products = pickle_load(config.products_path)

    order_products_prior = pd.merge(order_products_prior, products, on='product_id', how='left')

    ud_feat = pd.DataFrame()
    ud_feat['ud_first_order'] = order_products_prior.groupby(["user_id", "department_id"])['order_number_before_last_order'].max()
    ud_feat['ud_last_order'] = order_products_prior.groupby(["user_id", "department_id"])['order_number_before_last_order'].min()
    ud_feat['ud_distinct_order_num'] = order_products_prior.groupby(['user_id', 'department_id']).order_id.nunique()
    ud_feat['ud_distinct_product_num'] = order_products_prior.groupby(['user_id', 'department_id'])['product_id'].nunique()

    feats = ['ud_first_order', 'ud_last_order', 'ud_distinct_order_num', 'ud_distinct_product_num']
    pickle_dump(ud_feat[feats], '{}/user_department_feat.pkl'.format(config.feat_folder))
    print('Done - user_department features')