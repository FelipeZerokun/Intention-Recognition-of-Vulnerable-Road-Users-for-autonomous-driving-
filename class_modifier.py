import pandas as pd
from pathlib import Path
import os
from project_utils import check_path, check_file, check_os_windows

class ClassModifier:

    def __init__(self, target_class_dir: str, origin_class_dir: str):
        self.target_class_dir = target_class_dir
        self.origin_class_dir = origin_class_dir

        check_path(self.target_class_dir)
        check_path(self.origin_class_dir)

        self.target_pedestrian_number = target_class_dir.split('/')[-2]
        self.origin_pedestrian_number = origin_class_dir.split('/')[-2]

        self.class_name = target_class_dir.split('/')[-3]

        print(f"Class name {self.class_name}. Target pedestrian number {self.target_pedestrian_number}. Origin pedestrian number {self.origin_pedestrian_number}.")



def main():
    dataset_dir = '/media/felipezero/T7 Shield/DATA/thesis/Videos/video_01/classes/walking/'

    target_class_dir = dataset_dir + 'pedestrian_1/'
    origin_class_dir = dataset_dir + 'pedestrian_4/'

    class_modifier = ClassModifier(target_class_dir, origin_class_dir)

if __name__ == '__main__':
    main()




