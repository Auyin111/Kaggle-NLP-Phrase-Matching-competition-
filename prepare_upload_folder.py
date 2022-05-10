from utils import copy_and_overwrite
import os
import shutil

if __name__ == '__main__':

    source_dir = r"C:\Users\auyin11\PycharmProjects\competition_patent"
    destination_dir = r"C:\Users\auyin11\PycharmProjects\competition_patent_upload"

    copy_and_overwrite(source_dir, destination_dir)
    shutil.rmtree(os.path.join(destination_dir, 'output'))
