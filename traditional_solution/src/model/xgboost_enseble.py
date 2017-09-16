import numpy as np
import pandas as pd
import xgboost as xgb

class Xgboost_Ensemble(object):

    def __init__(self, random_state):
        self.random_state = random_state

    def set_dataset(self, x_train, y_train, x_test):
        self.x = x_train
        self.x_test = x_test
        self.y = y_train

    def set_parameters(self, objective, eval_metric, eta, max_depth, subsample, min_child_weight,
    	               col_sample_bytree, num_round=10000, early_stopping_rounds=100):
        self.params = {
            'objective': objective,
            'eval_metric': eval_metric,
            'eta': eta,
            'max_depth': max_depth,
            'subsample': subsample,
            'min_child_weight': min_child_weight,
            'col_sample_bytree': col_sample_bytree
        }
        self.num_round = num_round
        self.early_stopping_rounds = early_stopping_rounds

    def train(self, test_size=0.25, shuffle=False, stratify=None)
        print('Fitting the partial train set...')
        x_train, x_val, y_train, y_val = train_test_split(self.x, self.y, 
                test_size=test_size, shuffle=shuffle, stratify=stratify, random_state=self.random_state)

        xg_train = xgb.DMatrix(x_train, label=y_train)
        xg_val = xgb.DMatrix(x_val, label=y_val)
        watchlist  = [(xg_train,'train'), (xg_val,'eval')]
        model = xgb.train(self.params, xg_train, self.num_round, watchlist, early_stopping_rounds=self.early_stopping_rounds)

        print('Fitting the entire train set...')
        rounds = model.best_iteration
        self.model = xgb.train(params, xgb.DMatrix(self.x, label=self.y), rounds)
        print('Done')

    def predict(self):
        if self.model == None:
            return
        y_test_pred = self.model.predict(xgb.DMatrix(self.x_test))
        return y_test_pred
        