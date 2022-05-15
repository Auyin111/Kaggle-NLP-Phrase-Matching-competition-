import os
import shutil
from utils import cp_child_content


if __name__ == '__main__':

    dir_source = r"C:\Users\auyin11\PycharmProjects\competition_patent"
    dir_destination = r"C:\Users\auyin11\PycharmProjects\competition_patent_upload"
    selected_version = 'v3.1.1'

    list_cp_content = ['model', 'own_dataset', 'train_predict', 'gen_data',
                       'cfg.py', 'main.py', 'utils.py']

    shutil.rmtree(dir_destination)

    cp_child_content(dir_source, dir_destination, list_cp_content)
    shutil.copytree(os.path.join(dir_source, 'output', selected_version),
                    os.path.join(dir_destination, 'output', selected_version))