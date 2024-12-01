import pandas as pd
import cv2
from pathlib import Path

import os

from project_utils import check_path, check_file, check_os_windows

class ClassAnalysis:

    def __init__(self, class_dir: str):
        self.class_dir = class_dir
        check_path(self.class_dir)

        self.class_name = class_dir.split('/')[-3]
        self.pedestrian_number = class_dir.split('/')[-2]
        self.check_pedestrian_number()


        self.csv_dir = class_dir + f'{self.pedestrian_number}.csv'

        if not check_file(self.csv_dir):
            raise FileNotFoundError(f'CSV file {self.csv_dir} not found.')


        self.class_data = pd.read_csv(self.csv_dir)

    def check_pedestrian_number(self):
        """ Check if the pedestrian number in the files in the folder is the same as the one in the folder name.
        """
        for files in Path(self.class_dir).iterdir():
            if files.is_file():
                pedestrian_num_in_file = files.name.split('.')[0].split('_')
                pedestrian_num_in_file = pedestrian_num_in_file[0] + "_" + pedestrian_num_in_file[1]
                if self.pedestrian_number not in files.name:

                    new_file_name = files.name.replace(pedestrian_num_in_file, self.pedestrian_number)
                    os.rename(self.class_dir + files.name, self.class_dir + new_file_name)

                else:
                    print(f"Pedestrian number {self.pedestrian_number} found in file {files.name}.")

    def check_images_with_timestamp(self):
        """ Loop through each row and get the timestamp.
        Using the timestamp, find the images with the same timestamp in the folder.
        """
        for index, row in self.class_data.iterrows():
            timestamp = row['timestamp']
            image_path = self.find_image_with_timestamp(timestamp)

            if image_path is None:
                print(f"Image with timestamp {timestamp} not found.")
                continue

            image = cv2.imread(image_path)
            cv2.imshow('Image', image)
            print("Is the image correct? (y/n)")

            key = cv2.waitKey(0)
            if key == ord('n'):
                # Delete the row with the wrong timestamp and the delete the image from the folder
                cv2.destroyAllWindows()
                self.class_data.drop(index, inplace=True)
                print(f"Row with timestamp {timestamp} deleted.")
                os.remove(image_path)
                continue

    def find_image_with_timestamp(self, timestamp):
        """ The images in the folder have the following name format: pedestrian_n_timestamp.png
        Find the image with the same timestamp as the one in the row.
        """
        print(f"Checking image with timestamp {timestamp}")
        for image in Path(self.class_dir).iterdir():
            if image.is_file() and image.suffix == '.png':
                if str(timestamp) in image.name:
                    print(f"Image {image.name} found.")
                    image_path = self.class_dir + image.name
                    return image_path

        return None
def main():
    dataset_dir = '/media/felipezero/T7 Shield/DATA/thesis/Videos/video_01/classes/'
    class_name = 'standing_still/pedestrian_9/'

    class_analysis = ClassAnalysis(dataset_dir + class_name)
    # class_analysis.check_images_with_timestamp()


if __name__ == '__main__':
    main()