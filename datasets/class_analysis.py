import pandas as pd
import cv2
from pathlib import Path

import os

from project_utils.project_utils import check_path, check_file, check_os_windows

class ClassAnalysis:
    """
        Utility class to verify the consistency and integrity of labeled pedestrian class data.

        It ensures that:
            - The folder structure and file names match the expected pedestrian ID format.
            - All image frame timestamps match the ones recorded in the corresponding CSV file.
            - Any discrepancies (e.g., extra images or incorrect file names) are automatically corrected or flagged.

        Attributes:
            class_dir (str): Path to the directory containing images and CSV for a single pedestrian class.
            class_name (str): Name of the behavior class (e.g., 'walking_1').
            pedestrian_number (str): Pedestrian identifier extracted from the directory structure.
            csv_dir (str): Full path to the associated CSV file.
            class_data (pd.DataFrame): Loaded content of the CSV file with timestamps and metadata.
        """

    def __init__(self, class_dir: str):
        """
       Initializes the ClassAnalysis object, verifies pedestrian number consistency,
       loads the corresponding CSV, and checks image-to-CSV alignment.

       Args:
           class_dir (str): Path to the directory containing pedestrian images and CSV file.

       Raises:
           FileNotFoundError: If the expected CSV file is not found in the given directory.
       """

        self.class_dir = class_dir

        self.class_name = class_dir.split('/')[-3]
        self.pedestrian_number = class_dir.split('/')[-2]
        self.check_pedestrian_number()

        self.csv_dir = class_dir + f'{self.pedestrian_number}.csv'

        if not check_file(self.csv_dir):
            raise FileNotFoundError(f'CSV file {self.csv_dir} not found.')

        self.class_data = pd.read_csv(self.csv_dir)

        self.check_images_in_folder()

    def check_pedestrian_number(self):
        """
        Verifies that all filenames in the folder contain the correct pedestrian number
        (matching the directory name). If discrepancies are found, filenames are automatically
        corrected to reflect the correct pedestrian number.

        Example:
            If directory is named 'pedestrian_02', files should follow 'pedestrian_02_*.png' format.
        """
        for files in Path(self.class_dir).iterdir():
            if files.is_file():
                pedestrian_num_in_file = files.name.split('.')[0].split('_')
                pedestrian_num_in_file = pedestrian_num_in_file[0] + "_" + pedestrian_num_in_file[1]
                if self.pedestrian_number not in files.name:

                    new_file_name = files.name.replace(pedestrian_num_in_file, self.pedestrian_number)
                    os.rename(self.class_dir + files.name, self.class_dir + new_file_name)


    def check_images_in_folder(self):
        """
        Compares all image files in the folder to the timestamps listed in the CSV file.
        - Prints warnings for images not found in the CSV.
        - Reports total mismatches between image count and CSV row count.
        - If a CSV is found, updates the 'track_id' column to reflect the correct pedestrian number in the folder.
        - Provides a final status message on consistency.
        - Only checks PNG files and ignores other file types in the directory.

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
            elif image.is_file() and image.suffix == '.csv':
                #modify the track_id column in the CSV file
                csv_data = pd.read_csv(self.csv_dir)
                csv_data['track_id'] = int(self.pedestrian_number.split('_')[-1])
                csv_data.to_csv(self.csv_dir, index=False)

        if total_images != timestamp_data_len:
            print(f"Total images in the folder: {total_images}. Total images in the CSV file: {timestamp_data_len}.")
            images_ok = False

        if images_ok:
            print("All images in the folder are in the CSV file.")
        else:
            print("Images in the folder are not the same as the ones in the CSV file.")

    def check_images_with_timestamp(self):
        """
        Iterates through each row in the CSV and checks whether a corresponding image
        with the same timestamp exists in the folder.

        If an image is missing or visually incorrect (as judged by the user):
            - The row in the CSV is removed.
            - The image file is deleted from disk.

        Displays each image with OpenCV and prompts user input to validate.

        User input:
            - 'y': Keep image
            - 'n': Delete image and corresponding CSV row
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

        # # Save the new CSV file
        # self.class_data.to_csv(self.csv_dir, index=False)

    def find_image_with_timestamp(self, timestamp):
        """
        Searches for an image file in the folder that contains the given timestamp
        in its filename.

        Args:
            timestamp (int): Timestamp value to search for in image filenames.

        Returns:
            str or None: Full path to the matching image if found, otherwise None.
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
    # dataset_dir = '/media/felipezero/T7 Shield/DATA/thesis/intent_prediction_dataset/classes_01/'
    dataset_dir = 'E:/DATA/thesis/intent_prediction_dataset/classes_02/'
    class_name = 'walking_1/pedestrian_10/'

    class_analysis = ClassAnalysis(dataset_dir + class_name)
    class_analysis.check_images_with_timestamp()


if __name__ == '__main__':
    main()