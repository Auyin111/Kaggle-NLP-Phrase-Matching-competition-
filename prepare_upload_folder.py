import os
import shutil
from utils import cp_child_content
from pathlib import Path


if __name__ == '__main__':

    # TODO: use reference path
    dir_source = os.getcwd()
    dir_destination = os.path.join(os.path.dirname(dir_source), 'competition_patent_upload')

    list_model_version = ['deberta large', 'albert-base-v2', 'deberta-v3-base ver1']

    list_cp_content = ['model', 'own_dataset', 'train_predict', 'gen_data',
                       'cfg.py', 'main.py', 'utils.py', 'kaggle',
                       'ensemble', 'em_main.py', 'em1.0.7',
                       'paperspace_setup.sh', 'linux_pantent_requirement.txt', 'run_main.sh']

    Path(dir_source).mkdir(parents=True, exist_ok=True)
    shutil.rmtree(dir_destination)

    cp_child_content(dir_source, dir_destination, list_cp_content)
    for version in list_model_version:
        shutil.copytree(os.path.join(dir_source, 'output', version),
                        os.path.join(dir_destination, 'output', version))