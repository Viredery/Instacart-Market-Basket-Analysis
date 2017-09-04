import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, train_test_split


class LightgbmClassifier(object):

    def __init__(self, random_state):
        self.random_state = random_state

    def set_dataset(self, x_train, y_train, x_test, feature_name=None, categorical_feature_name=None):
        self.x = x_train[feature_name] if feature_name != None else x_train
        self.x_test = x_test[feature_name] if feature_name != None else x_test
        self.y = y_train

        self.categorical_feature_name = categorical_feature_name

    def set_parameters(self, boosting_type, objective, metric, learning_rate, num_leaves, max_depth, 
                       feature_fraction, bagging_fraction, bagging_freq, num_round=10000, early_stopping_rounds=100):
        self.params = {
            'boosting_type': boosting_type,
            'objective': objective,
            'metric': metric,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'max_depth': max_depth,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq
        }
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds

    def train_holdout(self, test_size=0.25, shuffle=False, stratify=None):
        print('Fitting 75 percent of train set...')
        x_train, x_valid, y_train, y_valid = train_test_split(self.x, self.y, 
                test_size=0.25, shuffle=False, stratify=None, random_state=self.random_state)
        d_train = lgb.Dataset(x_train, label=y_train, categorical_feature=self.categorical_feature_name)
        d_valid = lgb.Dataset(x_valid, label=y_valid, categorical_feature=self.categorical_feature_name)

        model = lgb.train(self.params, d_train, self.num_round, valid_sets=d_valid,
                          early_stopping_rounds=self.early_stopping_rounds, verbose_eval=10)        
        print('Fitting the entire train set...')
        rounds = model.best_iteration
        d = lgb.Dataset(self.x, label=self.y, categorical_feature=self.categorical_feature_name)
        self.model = lgb.train(self.params, d, rounds, verbose_eval=10)
        print('Done')

    def predict(self):
        y_test_pred = self.model.predict(self.x_test)
        return y_test_pred

    def cv_to_ensemble(self, n_splits=5, shuffle=False):
        y_train_score = np.zeros(self.x.shape[0])
        y_test_score = np.zeros((self.x_test.shape[0], n_splits))

        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_state)
        for i, train_idx, valid_idx in enumerate(kf.split(X, y)):
            print('CV: {}/{}...'.format(i+1, n_splits))
            x_train, x_valid = self.x.values[train_idx], self.x.values[valid_idx]
            y_train, y_valid = self.y[train_idx], self.y[valid_idx]

            d_train = lgb.Dataset(x_train, label=y_train, categorical_feature=self.categorical_feature_name)
            d_valid = lgb.Dataset(x_valid, label=y_valid, categorical_feature=self.categorical_feature_name)

            model = lgb.train(self.params, d_train, self.num_round, valid_sets=d_valid,
                              early_stopping_rounds=self.early_stopping_rounds, verbose_eval=10)
            y_train_score = model.predict(d_valid, num_iteration=model.best_iteration)
            y_train_pred[valid_idx] = y_train_pred

            y_test_score = model.predict(self.x_test, num_iteration=model.best_iteration)
            y_test_pred[:, i] = y_test_pred
        print('Done')
        return y_train_pred, np.mean(y_test_pred, axis=1)

    def evaluation(self, y_true, y_pred):
        return log_loss(y_true, y_pred)