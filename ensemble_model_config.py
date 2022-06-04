from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


class EnsembleModelConfig:

    def __init__(self, meta_learner, is_debug, n_jobs=20):

        self.meta_learner = meta_learner
        self.is_debug = is_debug
        self.n_jobs = n_jobs

        if meta_learner == 'lr':
            self._load_lr_config(meta_learner)
        else:
            raise Exception('can not find that config')

    def _load_lr_config(self, model_symbol):

        from sklearn.linear_model import LogisticRegression

        self.model_symbol = 'lr'
        self.pipe = Pipeline(
            [
                (
                    "scalar",
                    # TODO
                    RobustScaler(
                        quantile_range=('2.5, 97.5'),
                        with_centering=False,
                        with_scaling=False,
                    ),
                ),
                (self.model_symbol, LogisticRegression(n_jobs=self.n_jobs)),
            ]
        )

        # model parameters
        if self.is_debug:
            self.dict_parmas_grid = {
                f'{model_symbol}__C': [2, 3],
                f'{model_symbol}__fit_intercept': [True, False],
                f'{model_symbol}__penalty': ['l2'],
            }
        else:
            self.dict_parmas_grid = {
                f'{model_symbol}__C': [0.5, 1, 2, 3],
                f'{model_symbol}__fit_intercept': [True, False],
                f'{model_symbol}__penalty': ['l2', 'l1'],
            }

        # grid search
        self.dict_plot_gs = {
            "list_sample": ['train', 'test'],
            "list_discrete_parmas": [f'{model_symbol}__penalty', f'{model_symbol}__fit_intercept'],
            "list_const_parmas": [f'{model_symbol}__C'],
            "dict_scorer": {
                "neg_mean_absolute_error": "neg_mean_absolute_error",
                "neg_mean_squared_error": "neg_mean_squared_error",
                "r2": "r2",
            },
        }
        self.selected_scorer = "neg_mean_squared_error"
        self.return_train_score = True