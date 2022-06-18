import pandas as pd
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.model_selection import StratifiedShuffleSplit


class EnsembleModel:

    def __init__(self, df_valid, is_debug):

        self.df_valid = df_valid
        self.is_debug = is_debug



