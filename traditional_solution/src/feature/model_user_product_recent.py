import sys
import numpy as np
import pandas as pd

sys.path.append('../')
from param_config import config
from utils import pickle_load, pickle_dump

if __name__ == '__main__':
    print('Generating user_product features with user recent orders...')
    order_products_prior = pickle_load(config.order_products_prior_path)
    recent_orders = pickle_load(config.orders_path)[['user_id', 'order_id']].groupby(
            'user_id').tail(5 + 1)
    recent_order_products_prior = pd.merge(recent_orders, order_products_prior, on=['user_id', 'order_id'])

    up_feat = pd.DataFrame()
    up_feat['up_order_num'] = order_products_prior.groupby(['user_id', 'product_id']).size()
    up_feat['up_order_num_recent'] = recent_order_products_prior.groupby(['user_id', 'product_id']).size()

    up_feat['up_first_order_days_recent'] = recent_order_products_prior.groupby(['user_id', 'product_id']).order_days_before_last_order.max()
    up_feat['up_last_order_days_recent'] = recent_order_products_prior.groupby(['user_id', 'product_id']).order_days_before_last_order.min()
    up_feat['up_average_order_days_distance_recent'] = (up_feat['up_first_order_days_recent'] - up_feat['up_last_order_days_recent']) \
            / (up_feat['up_order_num_recent'] - 1)

    up_feat['up_last_order_days_ratio_recent'] = up_feat['up_last_order_days_recent'] / up_feat['up_average_order_days_distance_recent']
    up_feat['up_last_order_days_diff_recent'] = up_feat['up_last_order_days_recent'] - up_feat['up_average_order_days_distance_recent']

    # up_order_streak_recent
    def gen_up_order_streak(df):
        tmp = df.copy()
        tmp.user_id = 1

        up = tmp.pivot(index="product_id", columns='order_number').fillna(-1)
        up.columns = up.columns.droplevel(0)

        x = np.abs(up.diff(axis=1).fillna(2)).values[:, ::-1]
        df.set_index("product_id", inplace=True)
        df['up_order_streak_recent'] = np.multiply(np.argmax(x, axis=1) + 1, up.iloc[:, -1])
        df.reset_index(drop=False, inplace=True)
        return df

    recent_order_products_prior = recent_order_products_prior[['user_id', 'product_id', 'order_number']]
    streak = recent_order_products_prior.groupby('user_id').apply(gen_up_order_streak)
    streak = streak.drop("order_number", axis=1).drop_duplicates().reset_index(drop=True)
    streak = streak[['user_id', 'product_id', 'up_order_streak_recent']]

    up_feat = pd.merge(up_feat, streak, left_index=True, right_on=['user_id', 'product_id'], how='left').set_index(
            ['user_id', 'product_id'])

    # Fillna
    up_feat['up_order_num_recent'].fillna(0, inplace=True) 
    up_feat['up_first_order_days_recent'].fillna(999, inplace=True)
    up_feat['up_last_order_days_recent'].fillna(999, inplace=True)
    up_feat['up_average_order_days_distance_recent'].fillna(999, inplace=True)
    up_feat['up_last_order_days_ratio_recent'].fillna(999, inplace=True)
    up_feat['up_last_order_days_diff_recent'].fillna(999, inplace=True)
    up_feat['up_order_streak_recent'].fillna(-5, inplace=True)

    feats = ['up_order_num_recent', 'up_first_order_days_recent', 'up_last_order_days_recent',
             'up_average_order_days_distance_recent', 'up_last_order_days_ratio_recent',
             'up_last_order_days_diff_recent', 'up_order_streak_recent']
    pickle_dump(up_feat[feats], '{}/user_product_recent_feat.pkl'.format(config.feat_folder))
    print('Done - user_product features with user recent orders')

