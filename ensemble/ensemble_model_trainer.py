import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from ensemble.gs_analysis import all_gs_cvc
from utils import get_score
from ensemble.ensemble_model_config import EnsembleModelConfig
from sklearn.model_selection import train_test_split


class EnsembleModelTrainer:

    def __init__(self, list_model_version, list_em, em_version, is_debug):

        self.list_model_version = list_model_version
        self.list_em = list_em
        self.em_version = em_version
        self.is_debug = is_debug

        self.df_combined_valid = self.get_df_combined_valid(list_model_version)

    def _train_test_split(self):
        self.df_train_combined_valid, self.df_test_combined_valid = train_test_split(
            self.df_combined_valid,
            test_size=0.20, random_state=42,
            stratify=self.df_combined_valid.fold)

    def find_best_em(self):

        self._train_test_split()

        for em in self.list_em:
            em_cfg = EnsembleModelConfig(em, self.em_version, self.is_debug)

            grid = GridSearchCV(
                em_cfg.pipe,
                em_cfg.dict_parmas_grid,
                verbose=1,
                scoring=em_cfg.dict_plot_gs['dict_scorer'],
                cv=em_cfg.cv,
                return_train_score=em_cfg.return_train_score,
                refit=em_cfg.selected_scorer,
                n_jobs=10
            )

            grid = grid.fit(self.df_train_combined_valid[self.list_model_version],
                            y=self.df_train_combined_valid['score'],
                            groups=self.df_train_combined_valid['fold'])

            all_gs_cvc(em_cfg.dict_plot_gs, grid.cv_results_, grid.best_params_, grid.best_score_,
                       dir_html=os.path.join('output', em_cfg.em_version))

            self.df_test_combined_valid.loc[:, em] = grid.predict(
                self.df_test_combined_valid[self.list_model_version])

        self.__compare_perf()


    def __compare_perf(self):

        df_perf = pd.DataFrame()

        for  model in self.list_model_version + self.list_em:

            score = get_score(self.df_test_combined_valid['score'],
                              self.df_test_combined_valid[model])
            df_perf_spec = pd.DataFrame({'model':[model],
                                         'score':[score]})

            df_perf = pd.concat([df_perf, df_perf_spec])

        df_perf.reset_index(drop=True, inplace=True)
        print(df_perf)

    @staticmethod
    def get_df_combined_valid(list_version):

        df_combined_valid = pd.DataFrame()

        for version in list_version:
            df_spec_valid = pd.read_csv(os.path.join('output', version, 'df_valid.csv'))
            df_spec_valid = df_spec_valid.assign(version=version)

            df_combined_valid = pd.concat([df_combined_valid, df_spec_valid])

        df_combined_valid = df_combined_valid.pivot(index=['id', 'fold', 'score'], columns=['version'],
                                                    values='pred').reset_index()

        return df_combined_valid