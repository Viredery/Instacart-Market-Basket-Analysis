import numpy as np
import pandas as pd

from param_config import config
from utils import pickle_dump

if __name__ == '__main__':

    #################### orders ####################

    orders = pd.read_csv(config.raw_orders_path)

    orders.rename(columns={'days_since_prior_order': 'order_days_since_prior_order'}, inplace=True)
    orders['unrevised_days_before_last_order'] = orders.fillna(0).sort_values(
            'order_number', ascending=False).groupby('user_id')['order_days_since_prior_order'].cumsum()
    orders['order_days_before_last_order'] = orders.groupby(
            'user_id')['unrevised_days_before_last_order'].shift(-1).fillna(0).astype(np.int)
    del orders['unrevised_days_before_last_order']

    orders['order_number_before_last_order'] = orders.groupby('user_id')['order_number'].transform(max) - orders['order_number']

    pickle_dump(orders, config.orders_path)

    #################### products ####################

    products = pd.read_csv(config.raw_products_path)
    aisles = pd.read_csv(config.raw_aisles_path)
    departments = pd.read_csv(config.raw_departments_path)

    products = pd.merge(products, aisles, on='aisle_id', how='left')
    products = pd.merge(products, departments, on='department_id', how='left')

    pickle_dump(products[['product_id', 'aisle_id', 'department_id']], config.products_path)

    #################### order_products_prior ####################

    order_feats = ['order_id', 'user_id', 'order_number', 'order_days_since_prior_order',
                   'order_days_before_last_order', 'order_number_before_last_order']

    order_products_prior = pd.read_csv(config.raw_order_products_prior_path)
    order_products_prior = pd.merge(order_products_prior, orders[order_feats].copy(), on='order_id', how='left')

    pickle_dump(order_products_prior, config.order_products_prior_path)

    #################### order_products_train ####################

    order_products_train = pd.read_csv(config.raw_order_products_train_path)
    order_products_train = pd.merge(order_products_train, orders[order_feats].copy(), on='order_id', how='left')
    
    pickle_dump(order_products_train, config.order_products_train_path)
