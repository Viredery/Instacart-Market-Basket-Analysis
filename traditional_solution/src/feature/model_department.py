import sys
import numpy as np
import pandas as pd

sys.path.append('../')
from param_config import config
from utils import pickle_load, pickle_dump

if __name__ == '__main__':
    print('Generating department features...')
    order_products_prior = pickle_load(config.order_products_prior_path)
    products = pickle_load(config.products_path)
    order_products_prior = pd.merge(order_products_prior, products, on='product_id', how='left')

    department_feat = pd.DataFrame()
    department_feat['department_order_num'] = order_products_prior.groupby('department_id').size()
    department_feat['department_reorder_num'] = order_products_prior.groupby('department_id')['reordered'].sum()
    department_feat['department_reorder_ratio'] = department_feat['department_reorder_num'] / department_feat['department_order_num']

    department_feat['department_average_add_to_cart_order'] = order_products_prior.groupby('department_id')['add_to_cart_order'].mean()

    feats = ['department_order_num', 'department_reorder_num', 'department_reorder_ratio', 'department_average_add_to_cart_order']
    pickle_dump(department_feat[feats], '{}/department_feat.pkl'.format(config.feat_folder))
    print('Done - department features')