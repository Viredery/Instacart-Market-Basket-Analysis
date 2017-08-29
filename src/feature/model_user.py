import numpy as np
import pandas as pd

from param_config import config
from utils import pickle_load, pickle_dump

if __name__ == '__main__':
    print('Generating user features...')
    orders = pickle_load(config.orders_path)
    order_products_prior = pickle_load(config.order_products_prior_path)
    products = pickle_load(config.products_path)
    order_products_prior = pd.merge(order_products_prior, products, on='product_id', how='left')

    user_feat = pd.DataFrame()
    user_feat['user_order_num'] = orders[orders.eval_set == 'prior'].groupby('user_id').order_number.max()
    user_feat['user_average_days_since_prior_order'] = orders[orders.eval_set == 'prior'].groupby(
            'user_id').order_days_since_prior_order.mean()

    user_feat['user_total_product_num'] = order_products_prior.groupby('user_id').size()
    user_feat['user_average_product_num'] = user_feat['user_total_product_num'] / user_feat['user_order_num']
    user_feat['user_distinct_product_num'] = order_products_prior.groupby('user_id').product_id.nunique()

    user_feat['user_reorder_ratio'] = order_products_prior.groupby('user_id').reordered.sum() / \
            order_products_prior[order_products_prior.order_number != 1].groupby('user_id').size()

    user_feat['user_department_num'] = order_products_prior.groupby('user_id').department_id.nunique()
    user_feat['user_aisle_num'] = order_products_prior.groupby('user_id').aisle_id.nunique()

    feats = ['user_order_num', 'user_average_days_since_prior_order', 'user_total_product_num',
             'user_average_product_num', 'user_distinct_product_num', 'user_reorder_ratio',
             'user_department_num', 'user_aisle_num']
    pickle_dump(user_feat[feats], '{}/user_feat.csv'.format(config.feat_folder))
    print('Done - user features')