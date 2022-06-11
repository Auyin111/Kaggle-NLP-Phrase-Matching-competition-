from ensemble.ensemble_model import EnsembleModel
import category_encoders as ce

if __name__ == '__main__':

    list_em = ['rf', 'en']
    em_version = 'em1.0.1'
    is_debug = True
    # encoder = ce.BinaryEncoder()
    encoder = None
    list_model_version = ['v3.1.1', 'albert-base-v2', 'deberta-v3-base ver1']
    n_jobs = 10

    em_trainer = EnsembleModel(list_model_version, list_em, em_version, encoder, is_debug, n_jobs=n_jobs)
    em_trainer.find_best_model()
    # em_trainer.refit_em_model(best_model, dict_em_best_params,  list_model_version)

    em_trainer.find_best_em()