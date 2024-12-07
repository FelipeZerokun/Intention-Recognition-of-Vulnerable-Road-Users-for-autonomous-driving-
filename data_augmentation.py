import os
import cv2
import numpy as np
import pandas as pd

from project_utils import check_path


class ClassAugmentation:

    def __init__(self, dataset_dir: str, transformations: list):
        self.dataset_dir = dataset_dir
        self.transformations = transformations

        check_path(self.dataset_dir)
        self.classes, self.total_classes = self.check_classes()

        print(self.classes)

        self.augment_data(self.classes[0])


    def check_classes(self):
        """ Check how many classes are in the dataset and return a list with the classes directories.

        """
        action_classes = []
        classes = os.listdir(self.dataset_dir)
        total_classes = 0

        for action_class in classes:
            class_dir = self.dataset_dir + action_class + '/'
            total_data = len(os.listdir(class_dir))
            total_classes += total_data
            class_data = [action_class, total_data, class_dir]
            action_classes.append(class_data)

        return action_classes, total_classes



    def augment_data(self, data):
        """ Iterate through the images in the folder and apply the transformations.
        """
        class_name = data[0]
        total_data = data[1]
        class_dir = data[2]

        pedestrian_counter = 1

        for pedestrian in os.listdir(class_dir):
            pedestrian_num = pedestrian_counter + self.total_classes
            class_to_augment = class_dir + pedestrian + '/'
            print(class_to_augment)
            for image in os.listdir(class_to_augment):
                if image.endswith('.png'):
                    image_path = class_to_augment + image
                    self.augment_frame(image_path, self.transformations, class_dir, pedestrian_num)

                if image.endswith('.csv'):
                    class_data = pd.read_csv(class_to_augment + image)

            self.copy_csv_file(class_data, pedestrian_num, class_dir)
            pedestrian_counter += len(self.transformations)


    def augment_frame(self, frame_dir, transformations, output_dir: str, pedestrian_count: int):
        """ Apply transformations to a frame.
        """
        frame_name = frame_dir.split('/')[-1].split('_')[-1]

        frame = cv2.imread(frame_dir)

        if 'flip' in transformations:
            output_flip_dir = output_dir + 'pedestrian_' + str(pedestrian_count) + '/'
            check_path(output_flip_dir, create=True)

            flip = cv2.flip(frame, 1)
            flip_name = 'pedestrian_' + str(pedestrian_count) + '_' + frame_name
            cv2.imwrite(output_flip_dir + flip_name, flip)
            pedestrian_count += 1

        if 'brightness' in transformations:
            output_brightness_dir = output_dir + 'pedestrian_' + str(pedestrian_count) + '/'
            check_path(output_brightness_dir, create=True)

            value = 15
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hsv[:, :, 2] += cv2.add(hsv[:, :, 2], value)
            color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            color_name = 'pedestrian_' + str(pedestrian_count+1) + '_' + frame_name
            cv2.imwrite(output_brightness_dir + color_name, color)

            pedestrian_count += 1

        if 'rotate' in transformations:
            output_rotate_dir = output_dir + 'pedestrian_' + str(pedestrian_count + 1) + '/'
            check_path(output_rotate_dir, create=True)

            rows, cols, _ = frame.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
            rotate = cv2.warpAffine(frame, M, (cols, rows))
            rotate_name = 'pedestrian_' + str(pedestrian_count) + '_' + frame_name
            cv2.imwrite(output_rotate_dir + rotate_name, rotate)

            pedestrian_count += 1

    def copy_csv_file(self, dataframe, pedestrian_count, output_dir):

        for t in self.transformations:
            if t == 'flip':
                output_flip_dir = output_dir + 'pedestrian_' + str(pedestrian_count) + '/'
                dataframe.to_csv(output_flip_dir + f'pedestrian_{pedestrian_count}.csv', index=False)
                pedestrian_count += 1

            elif t == 'rotate':
                output_rotate_dir = output_dir + 'pedestrian_' + str(pedestrian_count) + '/'
                dataframe.to_csv(output_rotate_dir + f'pedestrian_{pedestrian_count}.csv', index=False)

                pedestrian_count += 1

            elif t == 'brightness':
                output_brightness_dir = output_dir + 'pedestrian_' + str(pedestrian_count) + '/'
                dataframe.to_csv(output_brightness_dir + f'pedestrian_{pedestrian_count}.csv', index=False)

                pedestrian_count += 1


def main():
    dataset_dir = '/media/felipezero/T7 Shield/DATA/thesis/action_recognition_dataset/'

    transformations = ['flip', 'brightness']  # 'flip', 'brightness' and/or 'rotate'

    class_augmentation = ClassAugmentation(dataset_dir, transformations)


if __name__ == '__main__':
    main()