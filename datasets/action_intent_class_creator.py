import os
import shutil
import pandas as pd
import cv2
from project_utils.project_utils import check_path, check_os_windows
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
            intent_action_dir = os.path.join(action_dir, 'intent')
            no_intent_action_dir = os.path.join(action_dir, 'no_intent')
            check_path(intent_action_dir, create=True)
            check_path(no_intent_action_dir, create=True)
            for action_number in os.listdir(action_dir):
                yield os.path.join(action_dir, action_number), intent_action_dir, no_intent_action_dir

    def annotate_intention(self):
        for action_number_dir, intent_dir, no_intent_dir in self._action_number_dirs():
            files_in_action_folder = os.listdir(action_number_dir)
            action = action_number_dir.split('/')[-2]
            pedestrian = action_number_dir.split('/')[-1]
            check_intent = False
            print(f"annotating for action {action} for pedestrian {pedestrian}")
            skip = input("Would you like to SKIP this pedestrian? (y/n): ").lower()
            if skip == 'y':
                continue
            # Perform labeling here
            for file in files_in_action_folder:
                if file.endswith('.csv'):
                    frame_data = pd.read_csv(os.path.join(action_number_dir, file))
                    # self._watch_frames(frame_data)
                if file.endswith('.png'):
                    frame = cv2.imread(os.path.join(action_number_dir, file))
                    cv2.imshow('Frame viewer', frame)
                    key = cv2.waitKey(0) & 0xFF
                    if key == 27:  # ESC key
                        break
                    if key == ord('d'):
                        continue

                    if key == ord('w'):
                        intent = 'walk'

            self._watch_frames(frame_data)
            while not check_intent:
                intent_from_user = input("Did the pedestrian intents to change their action? (y/n): ").lower()
                if intent_from_user == 'y':
                    if action == 'walking':
                        intent_label = 'stop_walking'
                        intent = 1
                        check_intent = True
                        for file in files_in_action_folder:
                            if file.endswith('.png'):
                                shutil.move(os.path.join(action_number_dir, file), os.path.join(no_intent_dir, file))

                    elif action == 'standing_still':
                        intent_label = 'start_walking'
                        intent = 1
                        check_intent = True
                        for file in files_in_action_folder:
                            if file.endswith('.png'):
                                shutil.move(os.path.join(action_number_dir, file), os.path.join(intent_dir, file))

                elif intent_from_user == 'n':
                    if action == 'walking':
                        intent_label = 'continue_walking'
                        intent = 0
                        check_intent = True
                        if file.endswith('.png'):
                            shutil.move(os.path.join(action_number_dir, file), os.path.join(no_intent_dir, file))

                    elif action == 'standing_still':
                        intent_label = 'continue_standing_still'
                        intent = 0
                        check_intent = True
                        if file.endswith('.png'):
                            shutil.move(os.path.join(action_number_dir, file), os.path.join(no_intent_dir, file))

                else:
                    print("Invalid input. Please enter 'y' or 'n'")



            # Add intent_label and intent as new colums to the frame_data
            frame_data['intent_label'] = intent_label
            frame_data['intent'] = intent

            # Save the new frame_data
            if intent == 0:
                file_name = os.path.join(intent_dir, "intent_data_" + pedestrian + ".csv")

            elif intent == 1:
                file_name = os.path.join(intent_dir, "intent_data_" + pedestrian + ".csv")

            frame_data.to_csv(os.path.join(file_name), index=False)

    def _watch_frames(self, frame_data):
        timestamps = list(frame_data['timestamp'])
        first_timestamp = timestamps[0]
        last_timestamp = timestamps[-1]

        frames = self._look_for_correct_video(first_timestamp, last_timestamp)
        color_frames = frames['rgb_frame'].tolist()
        depth_frames = frames['depth_frame'].tolist()
        timestamps = frames['timestamp'].tolist()

        index = 0
        while True:
            frame_path = color_frames[index]
            depth_path = depth_frames[index]
            frame_path = check_os_windows(frame_path)
            depth_path = check_os_windows(depth_path)

            frame = cv2.imread(frame_path)
            depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            # if values from the depth map are greater than 5000, make them 0


            cv2.imshow('Frame viewer', frame)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('r'):
                break

            if key == ord('a'):
                if index == 0:
                    print('Already at the first frame.')
                    continue
                else:
                    index -= 1
                    continue

            if key == ord('d'):
                if index == len(color_frames) - 1:
                    print('Already at the last frame.')
                    continue
                else:
                    index += 1
                    continue


    def _look_for_correct_video(self, first_frame, last_frame):
        video_01_path = os.path.join(self.frames_data_dir, 'video_01/navigation_data.csv')
        video_02_path = os.path.join(self.frames_data_dir, 'video_02/navigation_data.csv')
        video_03_path = os.path.join(self.frames_data_dir, 'video_03/navigation_data.csv')
        video_04_path = os.path.join(self.frames_data_dir, 'video_04/navigation_data.csv')

        videos_path = [video_01_path, video_02_path, video_03_path, video_04_path]
        for video_path in videos_path:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video {video_path} not found")

            frames_data = pd.read_csv(video_path)
            timestamps = list(frames_data['timestamp'])
            if first_frame in timestamps and last_frame in timestamps:
                # Return all the rows that are between the first and last frame
                return frames_data[(frames_data['timestamp'] >= first_frame)]

def main():
    frames_data_dir = '/media/felipezero/T7 Shield/DATA/thesis/Videos/'
    actions_dataset_dir = '/media/felipezero/T7 Shield/DATA/thesis/action_recognition_dataset/'
    output_dir = '/media/felipezero/T7 Shield/DATA/thesis/intent_prediction_dataset/'
    intent_prediction = IntentPredictionDataset(actions_dataset_dir, frames_data_dir, output_dir)

    intent_prediction.annotate_intention()



if __name__ == '__main__':
    main()