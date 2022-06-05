from ensemble.ensemble_model_trainer import EnsembleModelTrainer


if __name__ == '__main__':

    list_em = ['en']
    em_version = 'em1.0.1'
    is_debug = False
    list_model_version = ['v3.1.1', 'v1_tommy', 'v2_tommy']

    em_trainer = EnsembleModelTrainer(list_model_version, list_em, em_version, is_debug)

    em_trainer.find_best_em()