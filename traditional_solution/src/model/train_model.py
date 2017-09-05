import sys
import numpy as np
import pandas as pd

sys.path.append('../')
from param_config import config
from utils import pickle_dump, pickle_load

if __name__ == '__main__':

    x = pickle_load(, '{}/x_train_feat.pkl'.format(config.output_folder))
    x_test = pickle_load(, '{}/x_test_feat.pkl'.format(config.output_folder))
    y = pickle_load(, '{}/y_train.pkl'.format(config.output_folder))
