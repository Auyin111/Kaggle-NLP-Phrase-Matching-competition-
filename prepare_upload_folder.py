import os
from utils import rmtree, copy_and_overwrite


if __name__ == '__main__':

    source_dir = r"C:\Users\auyin11\PycharmProjects\competition_patent"
    destination_dir = r"C:\Users\auyin11\PycharmProjects\competition_patent_upload"

    list_folder_remove = ['output', '.git', '.idea', '.ipynb_checkpoints', '__pycache__', 'wandb',
                          'requirements.txt', '.gitignore', 'testing_2.py', 'main_testing.ipynb', 'testing.ipynb',
                          'TODO.md', 'submission.csv', 'demo-code.ipynb']

    # remove original directory
    if os.path.exists(destination_dir):
        rmtree(destination_dir)
    # copy directory
    copy_and_overwrite(source_dir, destination_dir)

    # remove folder and file which don't need to upload
    for file in list_folder_remove:

        path = os.path.join(destination_dir, file)

        if os.path.isdir(path):
            rmtree(path)
        else:
            os.remove(path)


