import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from lightgbm_classifier import LightgbmClassifier
from f1_optimizer import create_products_faron

sys.path.append('../')
from param_config import config
from utils import pickle_dump, pickle_load

if __name__ == '__main__':

    x = pickle_load('{}/x_train_feat.pkl'.format(config.output_folder))
    x_test = pickle_load('{}/x_test_feat.pkl'.format(config.output_folder))
    y = pickle_load('{}/y_train.pkl'.format(config.output_folder))
    
    feats = ['aisle_order_num', 'aisle_reorder_num', 'aisle_reorder_ratio', 'aisle_average_add_to_cart_order',

             'department_order_num', 'department_reorder_num', 'department_reorder_ratio', 'department_average_add_to_cart_order',

             'order_hour_of_day', 'order_days_since_prior_order', 'order_weekend', 'order_hour_of_day_bin_id',
             'order_days_since_prior_order_ratio', 'order_days_since_prior_order_diff', 'order_number_reorder_ratio',
             'order_delta_day_diff', 'order_delta_hour_diff',

             'product_order_num', 'product_reorder_num', 'product_reorder_frequency',
             'product_first_order_num', 'product_first_reorder_num', 'product_reorder_ratio',
             'product_user_order_only_once_num', 'product_user_order_only_once_ratio',
             'product_average_user_reorder_num', 'product_average_add_to_cart_order',

             'product_id_vector_1', 'product_id_vector_2',

             'user_order_num', 'user_order_days', 'user_average_days_since_prior_order', 'user_total_product_num',
             'user_average_product_num', 'user_distinct_product_num', 'user_reorder_ratio',
             'user_department_num', 'user_aisle_num',

             'ua_first_order', 'ua_last_order', 'ua_distinct_order_num', 'ua_distinct_product_num',
             'ud_first_order', 'ud_last_order', 'ud_distinct_order_num', 'ud_distinct_product_num',

             'up_order_num', 'up_average_add_to_cart_order', 'up_first_order', 'up_last_order', 'up_average_order',
             'up_average_order_distance', 'up_order_skew', 'up_last_order_ratio', 'up_last_order_diff',
             'product_average_order_distance', 'up_last_order_ratio_based_on_product', 
             'up_first_order_days', 'up_last_order_days', 'up_average_order_days', 'up_average_order_days_distance',
             'up_order_days_skew', 'up_last_order_days_ratio', 'up_last_order_days_diff',
             'product_average_order_days_distance', 'up_last_order_days_ratio_based_on_product',
             'up_hour_of_day_distance', 'up_dow_distance',

             'up_order_num_ratio', 'up_order_num_proportion', 'up_average_add_to_cart_order_ratio',
             'up_first_order_proportion', 'up_last_order_proportion','up_average_order_proportion',
             'up_last_order_proportion_ratio', 'up_first_order_days_proportion', 'up_last_order_days_proportion',
             'up_average_order_days_proportion', 'up_last_order_days_proportion_ratio',

             'up_order_num_recent', 'up_first_order_days_recent', 'up_last_order_days_recent',
             'up_average_order_days_distance_recent', 'up_last_order_days_ratio_recent',
             'up_last_order_days_diff_recent', 'up_order_streak_recent'
            ]

    lc = LightgbmClassifier(config.random_seed)
    lc.set_dataset(x=x, y=y, x_test=x_test, feature_name=feats)

    lc.set_parameters(boosting_type='gbdt',
                      objective='binary',
                      metric='binary_logloss', 
                      learning_rate=0.1, 
                      num_leaves=205, 
                      max_depth=15, 
                      feature_fraction=0.9, 
                      bagging_fraction=0.7, 
                      bagging_freq=5, 
                      num_round=10000, 
                      early_stopping_rounds=50)

    y_train_pred, y_test_pred = lc.cv_to_ensemble()

    x_test['pred'] = y_test_pred
    pickle_dump(x_test[['order_id', 'product_id', 'pred']], '{}/lgb_cls_test_pred.pkl'.format(config.output_folder))

    data = x_test
    order_ids = set(data['order_id'].unique())
    data = data.loc[data.pred > 0.015, ['order_id', 'pred', 'product_id']]
    out = [create_products_faron(group) for name, group in tqdm(data.groupby(data.order_id))]
    data = pd.DataFrame(data=out, columns=['order_id', 'products'])

    data.order_id = data.order_id.astype(np.int32)
    data.to_csv('{}/submission.csv'.format(config.output_folder), index=False)


    