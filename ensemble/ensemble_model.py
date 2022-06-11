import pandas as pd
import os
import json
from sklearn.model_selection import GridSearchCV
from ensemble.gs_analysis import all_gs_cvc
from utils import get_score, highlight_string
from ensemble.ensemble_model_config import EnsembleModelConfig
from sklearn.model_selection import train_test_split
from joblib import dump, load
from pathlib import Path


class EnsembleModel:

    def __init__(self, list_model_version, list_em, em_version, encoder,
                 is_debug,
                 n_jobs=1):

        self.list_model_version = list_model_version
        self.list_em = list_em
        self.em_version = em_version
        self.encoder = encoder
        self.is_debug = is_debug
        self.n_jobs = n_jobs
        self.em_cfg = EnsembleModelConfig(self.em_version, self.is_debug)

        self.df_combined_valid = self._get_df_combined_valid(list_model_version, self.em_cfg.dir_output)
        self.df_combined_valid.fold = self.df_combined_valid.fold.astype(str)

        self.list_col_feat = None

    def _train_test_split(self):
        df_train_combined_valid, df_test_combined_valid = train_test_split(
            self.df_combined_valid,
            test_size=0.20, random_state=42,
            stratify=self.df_combined_valid.fold)

        return df_train_combined_valid, df_test_combined_valid

    # TODO: use nested cross validation
    def find_best_model(self):

        df_train_combined_valid, df_test_combined_valid = self._train_test_split()
        dict_all_em_best_params = {}

        for em in self.list_em:

            self.em_cfg.change_ml_cfg(em, self.encoder)
            if self.encoder is not None:
                self.list_col_feat = self.list_model_version + ['fold']
            else:
                self.list_col_feat = self.list_model_version

            print(self.list_col_feat)
            print(df_train_combined_valid.head())

            n_jobs = 1 if em == 'en' else self.n_jobs
            print(f'em: {em}, n_jobs: {n_jobs}')

            grid = GridSearchCV(
                self.em_cfg.pipe,
                self.em_cfg.dict_parmas_grid,
                verbose=1,
                scoring=self.em_cfg.dict_plot_gs['dict_scorer'],
                cv=self.em_cfg.cv,
                return_train_score=self.em_cfg.return_train_score,
                refit=self.em_cfg.selected_scorer,
                n_jobs=n_jobs
            )

            dir_em_model = os.path.join(self.em_cfg.dir_output, self.em_version, em)
            Path(dir_em_model).mkdir(parents=True, exist_ok=True)
            grid = grid.fit(df_train_combined_valid[self.list_col_feat],
                            y=df_train_combined_valid['score'],
                            groups=df_train_combined_valid['fold'])

            dict_all_em_best_params[em] = grid.best_params_
            dump(grid, os.path.join(dir_em_model, 'model.joblib'))

            all_gs_cvc(self.em_cfg.dict_plot_gs, grid.cv_results_, grid.best_params_, grid.best_score_,
                       dir_html=os.path.join(dir_em_model, 'grid'))

            df_train_combined_valid.loc[:, em] = grid.predict(df_train_combined_valid[self.list_col_feat])
            df_test_combined_valid.loc[:, em] = grid.predict(df_test_combined_valid[self.list_col_feat])

        df_perf_train = self.__compare_perf(df_train_combined_valid, dataset='em_train_set')
        df_perf_test = self.__compare_perf(df_test_combined_valid, dataset='em_test_set')

        df_perf = pd.concat([df_perf_train, df_perf_test])
        best_model = df_perf_test[df_perf_test.score == df_perf_test.score.max()].model.values[0]

        print(f'the model performance:\n'
              f'{df_perf}')
        if best_model in self.list_em:
            print(highlight_string((f'the {best_model} ensemble model have'
                                    f' the best performance so use it to make prediction')))
        else:
            # TODO: select the best individual model
            print(highlight_string(f'ensemble model is worse than the individual model '
                                   f'so we use {best_model} to make prediction'))
            raise Exception('TODO')

        dict_em_best_params = dict_all_em_best_params[best_model]

        # save best model and best parmas
        with open(os.path.join(self.em_cfg.dir_output, self.em_version, 'dict_best_model_n_parmas.json'), "w") as fp:
            json.dump({'best_model': best_model,
                       'dict_em_best_params': dict_em_best_params},
                      fp)

    def __compare_perf(self, df, dataset):

        df_perf = pd.DataFrame()

        for model in self.list_model_version + self.list_em:

            score = get_score(df['score'],
                              df[model])
            df_perf_spec = pd.DataFrame({'model': [model],
                                         'score': [score]})

            df_perf = pd.concat([df_perf, df_perf_spec])

        df_perf.reset_index(drop=True, inplace=True)
        df_perf.loc[:, 'dataset'] = dataset

        return df_perf

    # TODO: train a model without test set
    def refit_em_model(self, best_model, dict_em_best_params, list_model_version):
        pass
        # self.em_cfg.change_ml_cfg(best_model)
        # self.em_cfg.pipe.set_params(**dict_em_best_params)
        # self.em_cfg.pipe.fit(self.df_combined_valid[list_model_version],
        #                      self.df_combined_valid['score'])

    @staticmethod
    def _get_df_combined_valid(list_version, dir_out):

        df_combined_valid = pd.DataFrame()

        for version in list_version:
            df_spec_valid = pd.read_csv(os.path.join(dir_out, version, 'df_valid.csv'))
            df_spec_valid = df_spec_valid.assign(version=version)

            df_combined_valid = pd.concat([df_combined_valid, df_spec_valid])

        df_combined_valid = df_combined_valid.pivot(index=['id', 'fold', 'score'], columns=['version'],
                                                    values='pred').reset_index()

        return df_combined_valid