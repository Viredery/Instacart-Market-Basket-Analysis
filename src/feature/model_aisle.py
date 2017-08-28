import numpy as np
import pandas as pd

from param_config import config
from utils import pickle_load, pickle_dump

if __name__ == '__main__':
    print('Generating aisle features...')
    order_products_prior = pickle_load(config.order_products_prior_path)
    products = pickle_load(config.products_path)
    order_products_prior = pd.merge(order_products_prior, products, on='product_id', how='left')

    aisle_feat = pd.DataFrame()
    aisle_feat['aisle_order_num'] = order_products_prior.groupby('aisle_id').size()
    aisle_feat['aisle_reorder_num'] = order_products_prior.groupby('aisle_id')['reordered'].sum()
    aisle_feat['aisle_reorder_ratio'] = aisle_feat['aisle_reorder_num'] / aisle_feat['aisle_order_num']

    aisle_feat['aisle_average_add_to_cart_order'] = order_products_prior.groupby('aisle_id')['add_to_cart_order'].mean()

    feats = ['aisle_order_num', 'aisle_reorder_num', 'aisle_reorder_ratio', 'aisle_average_add_to_cart_order']
    pickle_dump(aisle_feat[feats], '{}/aisle_feat.pkl'.format(config.feat_folder))
    print('Done - aisle features')