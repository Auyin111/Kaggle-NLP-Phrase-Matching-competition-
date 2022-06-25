import os
import shutil
from utils import cp_child_content


if __name__ == '__main__':

    # TODO: use reference path
    dir_source = r"C:\Users\tommy\Desktop\Machine Learning\Kaggle\USPPPM\competition_patent"
    dir_destination = r"C:\Users\tommy\Desktop\Machine Learning\Kaggle\USPPPM\competition_patent_upload"
    if not os.path.isdir(dir_destination):  # Auto create destination dir if not yet exist
        os.mkdir(dir_destination)

    selected_version = None # "None" if train on Kaggle

    list_cp_content = ['model', 'own_dataset', 'train_predict', 'gen_data',
                       'cfg.py', 'main.py', 'utils.py', 'kaggle',
                       'lambda_labs_setup.sh', 'linux_pantent_requirement.txt', 'run_main.sh']

    shutil.rmtree(dir_destination)

    cp_child_content(dir_source, dir_destination, list_cp_content)
    if selected_version is not None:
        shutil.copytree(os.path.join(dir_source, 'output', selected_version),
                        os.path.join(dir_destination, 'output', selected_version))