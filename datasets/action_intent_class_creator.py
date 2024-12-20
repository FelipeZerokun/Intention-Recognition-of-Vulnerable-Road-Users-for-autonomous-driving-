import os

import pandas as pd
import cv2
from project_utils.project_utils import check_path
class IntentPredictionDataset:
    def __init__(self, actions_dataset_dir: str, frames_data_dir: str, output_dir: str):

        self._DATA_COLUMNS = ['timestamp']

        self.actions_dir = actions_dataset_dir
        self.frames_data_dir = frames_data_dir
        self.output_dir = output_dir
        check_path(self.output_dir, create=True)

        if not check_path(self.actions_dir):
            raise FileNotFoundError(f"Directory {self.actions_dir} not found")

        if not check_path(self.frames_data_dir):
            raise FileNotFoundError(f"Directory {self.frames_data_dir} not found")

        self.pedestrian_actions = self._action_counter()
        print(self.pedestrian_actions)


    def _action_counter(self):
        action_counter = {}
        for action in os.listdir(self.actions_dir):
            action_counter[action] = len(os.listdir(os.path.join(self.actions_dir, action)))

        return action_counter

    def check_correct_frames_data(self):
        """
        Check if the frames data CSV file has the correct columns.

        Raises:
            ValueError: If the columns are not correct.
        """
        frame_data = pd.read_csv(self.frames_data)

        columns = frame_data.columns

        for column in self._DATA_COLUMNS:
            if column not in columns:
                raise ValueError(f'Column {column} not found in the frames data CSV file.')

        return frame_data

    def _action_number_dirs(self):
        for action in os.listdir(self.actions_dir):
            action_dir = os.path.join(self.actions_dir, action)
            for action_number in os.listdir(action_dir):
                yield os.path.join(action_dir, action_number)

    def annotate_intention(self):
        for action_number_dir in self._action_number_dirs():
            files_in_action_folder = os.listdir(action_number_dir)
            print("annotating for action: ", action_number_dir)
            # Perform labeling here
            for file in files_in_action_folder:
                if file.endswith('.png'):
                    intent = 'unknown'
                    frame = cv2.imread(os.path.join(action_number_dir, file))
                    cv2.imshow('Frame viewer', frame)
                    key = cv2.waitKey(0) & 0xFF
                    if key == 27:  # ESC key
                        break
                    if key == ord('d'):
                        continue

                if intent == 'unknown':
                    intent = input("Enter the intention of the pedestrian: ")




def main():
    frames_data_dir = '/media/felipezero/T7 Shield/DATA/thesis/Videos/'
    actions_dataset_dir = '/media/felipezero/T7 Shield/DATA/thesis/action_recognition_dataset/'
    output_dir = '/media/felipezero/T7 Shield/DATA/thesis/intent_prediction_dataset/'
    intent_prediction = IntentPredictionDataset(actions_dataset_dir, frames_data_dir, output_dir)

    intent_prediction.annotate_intention()



if __name__ == '__main__':
    main()