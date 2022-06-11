from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer
from utils import get_score
from utilities.custom_splitter import StratifiedColShuffleSplit, StratifiedColKFold
import datetime
import os
from utils import highlight_string
from sklearn.compose import ColumnTransformer

corr_scorer = make_scorer(get_score, greater_is_better=True)


class EnsembleModelConfig:

    def __init__(self, em_version, is_debug):

        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        self.dir_output = os.path.join('output')
        self.dir_em_output = os.path.join('output', f"{em_version} - {self.current_time}")
        self.is_debug = is_debug

        self.cv = StratifiedColKFold(n_splits=5)
        self.meta_learner = None

    def change_ml_cfg(self, meta_learner, encoder):
        """change meta learner config"""

        print(highlight_string(f'meta_learner is changed from {self.meta_learner} to meta_learner'))
        self.meta_learner = meta_learner

        if meta_learner == 'en':
            self._load_en_config(meta_learner, encoder)
        elif meta_learner == 'rf':
            self._load_rf_config(meta_learner, encoder)
        else:
            raise Exception('can not find that config')

    def _load_en_config(self, model_symbol, encoder):

        from sklearn.linear_model import ElasticNet

        self.model_symbol = model_symbol

        list_pipeline = [('en', ElasticNet())]
        self.pipe = Pipeline(
            list_pipeline if encoder is None else self._assign_encoder_in_pipe(encoder, list_pipeline)
        )

        # model parameters
        if self.is_debug:
            self.dict_parmas_grid = {
                f'{model_symbol}__alpha': [0.5, 1],
                f'{model_symbol}__fit_intercept': [True, False],
                f'{model_symbol}__l1_ratio': [0, 1],
                f'{model_symbol}__max_iter': [100]
            }
        else:
            self.dict_parmas_grid = {
                f'{model_symbol}__alpha': [0.5, 0.75, 1, 2],
                f'{model_symbol}__fit_intercept': [True, False],
                f'{model_symbol}__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
                f'{model_symbol}__max_iter': [5000]
            }

        # grid search
        self.dict_plot_gs = {
            "list_sample": ['train', 'test'],
            "list_discrete_parmas": [f'{model_symbol}__fit_intercept'],
            "list_const_parmas": [f'{model_symbol}__l1_ratio', f'{model_symbol}__alpha'],
            "dict_scorer": {
                "neg_mean_absolute_error": "neg_mean_absolute_error",
                "neg_mean_squared_error": "neg_mean_squared_error",
                "r2": "r2",
                "corr_scorer": corr_scorer
            },
        }
        self.selected_scorer = "corr_scorer"
        self.return_train_score = True

    @staticmethod
    def _assign_encoder_in_pipe(encoder, list_pipeline):

        categorical_transformer = encoder
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", categorical_transformer, ['fold']),
            ]
        )

        list_pipeline.insert(0, ('preprocessor', preprocessor))

        return list_pipeline

    def _load_rf_config(self, model_symbol, encoder):

        from sklearn.ensemble import RandomForestRegressor

        self.model_symbol = model_symbol
        list_pipeline = [('rf', RandomForestRegressor())]

        self.pipe = Pipeline(
            list_pipeline if encoder is None else self._assign_encoder_in_pipe(encoder, list_pipeline)
        )

        # model parameters
        if self.is_debug:
            self.dict_parmas_grid = {
                f'{model_symbol}__n_estimators': [10],
                f'{model_symbol}__criterion': ['squared_error'],
                f'{model_symbol}__max_depth': [3, 4],
                f'{model_symbol}__min_samples_split': [2]
            }
        else:
            self.dict_parmas_grid = {
                f'{model_symbol}__n_estimators': [100],
                f'{model_symbol}__criterion': ['squared_error'],
                f'{model_symbol}__max_depth': [4, 5, 6, 7, 8],
                f'{model_symbol}__min_samples_split': [2, 4, 6]
            }

        # grid search
        self.dict_plot_gs = {
            "list_sample": ['train', 'test'],
            "list_discrete_parmas": [f'{model_symbol}__criterion'],
            "list_const_parmas": [f'{model_symbol}__n_estimators', f'{model_symbol}__max_depth',
                                  f'{model_symbol}__min_samples_split'],
            "dict_scorer": {
                "neg_mean_absolute_error": "neg_mean_absolute_error",
                "neg_mean_squared_error": "neg_mean_squared_error",
                "r2": "r2",
                "corr_scorer": corr_scorer
            },
        }
        self.selected_scorer = "corr_scorer"
        self.return_train_score = True