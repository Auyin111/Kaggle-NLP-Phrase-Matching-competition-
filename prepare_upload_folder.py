import os
import shutil
from utils import cp_child_content
from pathlib import Path


if __name__ == '__main__':

    # TODO: use reference path
    dir_source = os.getcwd()
    dir_destination = os.path.join(os.path.dirname(dir_source), 'competition_patent_upload')

    list_model_version = [
        'bert-for-patents_MSE_BS64_grp2short_v1',
        'deberta_base_MSE_BS64_grp2short_v1',
        # 'deberta_large_MSE_BS64_grp2short_v1',

        # 'roberta-large_MSE_BS64_grp2short_v1',
        'deberta_large_MSE_BS64_grp2short_v2head_E12',

        'albert-base-v2',

        'em1.0.17']

    list_cp_content = ['model', 'own_dataset', 'train_predict', 'gen_data',
                       'cfg.py', 'main.py', 'utils.py', 'kaggle',
                       'ensemble', 'em_main.py',
                       'paperspace_setup.sh', 'linux_pantent_requirement.txt', 'run_main.sh']

    Path(dir_source).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dir_destination)

    cp_child_content(dir_source, dir_destination, list_cp_content)
    for version in list_model_version:
        shutil.copytree(os.path.join(dir_source, 'output', version),
                        os.path.join(dir_destination, 'output', version))