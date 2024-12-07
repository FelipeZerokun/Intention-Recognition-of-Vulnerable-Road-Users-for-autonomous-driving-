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

        self.check_images_in_folder()

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


    def check_images_in_folder(self):
        """ Check if the images in the folder are the same as the ones in the CSV file. If there are more images in the folder, with a different timestamp, add them to the CSV file.
        """
        images_ok = True
        total_images = 0
        timestamp_data = self.class_data['timestamp'].values
        timestamp_data_len = len(timestamp_data)
        for image in Path(self.class_dir).iterdir():
            if image.is_file() and image.suffix == '.png':
                timestamp = int(image.name.split('_')[-1].split('.')[0])
                if timestamp not in timestamp_data:
                    print(f"Image with timestamp {timestamp} not found in the CSV file.")
                    images_ok = False
                else:
                    total_images += 1
        if total_images != timestamp_data_len:
            print(f"Total images in the folder: {total_images}. Total images in the CSV file: {timestamp_data_len}.")
            images_ok = False

        if images_ok:
            print("All images in the folder are in the CSV file.")
        else:
            print("Images in the folder are not the same as the ones in the CSV file.")

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

        # Save the new CSV file
        self.class_data.to_csv(self.csv_dir, index=False)

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
    dataset_dir = '/media/felipezero/T7 Shield/DATA/thesis/Videos/video_04/classes/'
    class_name = 'walking/pedestrian_169/'

    class_analysis = ClassAnalysis(dataset_dir + class_name)
    class_analysis.check_images_with_timestamp()


if __name__ == '__main__':
    main()