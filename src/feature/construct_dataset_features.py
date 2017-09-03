import numpy as np
import pandas as pd

from param_config import config
from utils import pickle_load, pickle_dump

if __name__ == '__main__':

	print('Merging total features into train/test dataset...')

	print('Loading train/test dataset...')
    x_train = pickle_load('{}/x_train.pkl'.format(config.data_folder))
    x_test = pickle_load('{}/x_test.pkl'.format(config.data_folder))

    print('Loading features...')
    products = pickle_load(config.products_path)
    order_feat = pickle_load('{}/order_feat.pkl'.format(config.feat_folder))
    product_feat = pickle_load('{}/product_feat.pkl'.format(config.feat_folder))
    product_vector_feat = pickle_load('{}/product_vector_feat.pkl'.format(config.feat_folder))
    user_feat = pickle_load('{}/user_feat.pkl'.format(config.feat_folder))
    aisle_feat = pickle_load('{}/aisle_feat.pkl'.format(config.feat_folder))
    department_feat = pickle_load('{}/department_feat.pkl'.format(config.feat_folder))
    user_product_feat = pickle_load('{}/user_product_feat.pkl'.format(config.feat_folder))
    user_aisle_feat = pickle_load('{}/user_aisle_feat.pkl'.format(config.feat_folder))
    user_department_feat = pickle_load('{}/user_department_feat.pkl'.format(config.feat_folder))
    user_product_recent_feat = pickle_load('{}/user_product_recent_feat.pkl'.format(config.feat_folder))
    user_product_dependent_feat = pickle_load('{}/user_product_dependent_feat.pkl'.format(config.feat_folder))

    print('Merging...')


    def merge_features(df):
        df = pd.merge(df, products, on='product_id', how='left')

        df = pd.merge(df, order_feat, left_on='order_id', right_index=True, how='left')
        df = pd.merge(df, product_feat, left_on='product_id', right_index=True, how='left')
        df = pd.merge(df, product_vector_feat, left_on='product_id', right_index=True, how='left')
        df = pd.merge(df, user_feat, left_on='user_id', right_index=True, how='left')
        df = pd.merge(df, aisle_feat, left_on='aisle_id', right_index=True, how='left')
        df = pd.merge(df, department_feat, left_on='department_id', right_index=True, how='left')

        df = pd.merge(df, user_product_feat, left_on=['user_id', 'product_id'], right_index=True, how='left')
        df = pd.merge(df, user_product_recent_feat, left_on=['user_id', 'product_id'], right_index=True, how='left')
        df = pd.merge(df, user_product_dependent_feat, left_on=['user_id', 'product_id'], right_index=True, how='left')

        df = pd.merge(df, user_aisle_feat, left_on=['user_id', 'aisle_id'], right_index=True, how='left')
        df = pd.merge(df, user_department_feat, left_on=['user_id', 'department_id'], right_index=True, how='left')
        return df

    x_train_feat = merge_features(x_train)
    x_test_feat = merge_features(x_test)

    pickle_dump(x_train_feat, '{}/x_train_feat.pkl'.format(data_folder))
    pickle_dump(x_test_feat, '{}/x_test_feat.pkl'.format(data_folder))
    print('Done')

