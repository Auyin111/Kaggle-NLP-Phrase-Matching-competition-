from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import make_scorer
from utils import get_score
from utilities.custom_splitter import StratifiedColShuffleSplit, StratifiedColKFold

corr_scorer = make_scorer(get_score, greater_is_better=True)


class EnsembleModelConfig:

    def __init__(self, meta_learner, em_version, is_debug, n_jobs=20):

        self.meta_learner = meta_learner
        self.em_version = em_version
        self.is_debug = is_debug
        self.n_jobs = n_jobs

        if meta_learner == 'en':
            self._load_en_config(meta_learner)
        else:
            raise Exception('can not find that config')

        self.cv = StratifiedColKFold(n_splits=5)

    def _load_en_config(self, model_symbol):

        from sklearn.linear_model import ElasticNet

        self.model_symbol = model_symbol
        self.pipe = Pipeline(
            [
                # TODO: scalar
                # (
                #     "scalar",
                #     RobustScaler(
                #         quantile_range=('2.5, 97.5'),
                #         with_centering=True,
                #         with_scaling=False,
                #     ),
                # ),
                ('en', ElasticNet()),
            ]
        )

        # model parameters
        if self.is_debug:
            self.dict_parmas_grid = {
                f'{model_symbol}__alpha': [0.5, 1],
                f'{model_symbol}__fit_intercept': [True, False],
                f'{model_symbol}__l1_ratio': [0, 1],
                f'{model_symbol}__max_iter': [5000]
            }
        else:
            self.dict_parmas_grid = {
                f'{model_symbol}__alpha': [0.5, 1, 2],
                f'{model_symbol}__fit_intercept': [True, False],
                f'{model_symbol}__l1_ratio': [0, 0.25, 0.5, 0.75, 1],
                f'{model_symbol}__max_iter': [2000]
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