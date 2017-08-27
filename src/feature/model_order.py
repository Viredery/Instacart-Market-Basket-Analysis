import numpy as np
import pandas as pd

from param_config import config
from utils import pickle_load, pickle_dump

if __name__ == '__main__':
    print('Generating order features...')
    orders = pickle_load(config.orders_path)

    order_feat = orders[orders.eval_set != 'prior'].copy().reset_index(drop=True)

    order_feat['order_weekend'] = np.where(order_feat['order_dow'].isin([0, 6]), 1, 0)
    order_feat['order_hour_of_day_bin_id'] = np.where(order_feat['order_hour_of_day'].isin(np.arange(7,13)),
                                                      0,
                                                      np.where(order_feat['order_hour_of_day'].isin(np.arange(13,19)), 1, 2))

    order_feat['order_days_since_prior_order_ratio'] = \
            (orders[orders.eval_set != 'prior'].set_index('user_id')['order_days_since_prior_order'] / \
            orders[orders.eval_set == 'prior'].groupby('user_id')['order_days_since_prior_order'].mean()).values

    order_feat['order_days_since_prior_order_diff'] = \
            (orders[orders.eval_set != 'prior'].set_index('user_id')['order_days_since_prior_order'] - \
            orders[orders.eval_set == 'prior'].groupby('user_id')['order_days_since_prior_order'].mean()).values

    # Fillna
    # As for NaN in the feature 'order_days_since_prior_order_ratio', the numerator and the denominator are all zero. Hence fill 1
    order_feat['order_days_since_prior_order_ratio'].fillna(1, inplace=True)

    # Generate the feature based on order_number
    order_products_prior = pickle_load(config.order_products_prior_path)
    order_number_reorder_ratio = order_products_prior.groupby('order_number')['reordered'].mean().to_frame()
    order_number_reorder_ratio.columns = ['order_number_reorder_ratio']
    order_feat = pd.merge(order_feat, order_number_reorder_ratio, left_on='order_number', right_index=True, how='left')


    feats = ['order_dow', 'order_hour_of_day', 'order_days_since_prior_order', 'order_weekend', 'order_hour_of_day_bin_id',
             'order_days_since_prior_order_ratio', 'order_days_since_prior_order_diff', 'order_number_reorder_ratio']

    pickle_dump(order_feat[feats], '{}/order_feat.pkl'.format(config.feat_folder))
    print('Done - order features')