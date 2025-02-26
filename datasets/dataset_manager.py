from pathlib import Path
import shutil

import cv2
import pandas as pd

from project_utils.project_utils import (check_path, check_file, check_os_windows, estimate_pedestrian_distance, estimate_pedestrian_position, save_csv_file)

import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort


class DatasetManager:
    def __init__(self, data_path: str, output_folder: str):
        """
        Class to create a dataset with human actions and intent from several frames extracted from
        a video file or from rosbags.

        Args:
            video_data (Path): Path to the CSV file with the frame's data.

        """
        self._DATA_COLUMNS = ['timestamp', 'Robot odometry', 'rgb_frame', 'depth_frame']

        self.frames_data = Path(data_path, 'navigation_data.csv')
        self.camera_info = Path(data_path, 'camera_info.csv')

        self.source = data_path.split('/')[-2]

        self.camera_info = check_os_windows(self.camera_info)

        self.image_size, self.camera_intrinsics = self.get_camera_info()

        # Lad both the YoloV5 model and the DeepSort tracker
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        self.tracker = DeepSort(
            max_age=30,
            nn_budget=70,
            nms_max_overlap=1.0,
        )

        self.frames_data = check_os_windows(self.frames_data)

        assert check_file(self.frames_data) is True

        self.output_folder = check_os_windows(output_folder)
        check_path(folder_path=self.output_folder, create=True)

        self.pedestrian_counter = 0
        self.start_timestamp = 0

    def get_camera_info(self):
        """
        Get the camera info from the CSV file.
        """
        camera_info = pd.read_csv(self.camera_info)
        image_height = camera_info.loc[0, 'image_height']
        image_width = camera_info.loc[0, 'image_width']

        intrinsic_matrix_str = camera_info.loc[0, 'Intrinsic_camera_matrix']

        # Convert the string representation of the matrix to a numpy array
        intrinsic_matrix = np.fromstring(intrinsic_matrix_str.strip('[]'), sep=' ').reshape((3, 3))

        return ([image_height, image_width], intrinsic_matrix)

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

    def create_classes_for_action_recognition(self):
        """
        Manual creation of classes for the dataset for the Action Recognition model.
        """
        frame_data = self.check_correct_frames_data()
        frame_data = frame_data.loc[frame_data['timestamp'] >= self.start_timestamp]
        color_frames = frame_data['rgb_frame'].tolist()
        depth_frames = frame_data['depth_frame'].tolist()
        timestamps = frame_data['timestamp'].tolist()

        recording = False
        start_timestamp = 0
        end_timestamp = 0

        index = 0
        while True:
            timestamp = timestamps[index]
            frame_path = color_frames[index]
            depth_path = depth_frames[index]
            frame_path = check_os_windows(frame_path)
            depth_path = check_os_windows(depth_path)

            frame = cv2.imread(frame_path)
            depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            added_images = self.combine_rgb_depth_frames(frame, depth_map, depth_threshold=10000)

            cv2.imshow('Frame viewer', added_images)

            key = cv2.waitKey(0) & 0xFF
            if key == 27: #ESC key
                break

            if key == ord('r'):
                if not recording:
                    recording = True
                    start_timestamp = timestamps[index]
                    print(f'Start action recording at {start_timestamp}')
                    continue

                if recording:
                    recording = False
                    end_timestamp = timestamps[index]
                    print(f'Stop Recording at {end_timestamp}')

                    if start_timestamp >= end_timestamp:
                        print('End timestamp must be greater than start timestamp.')
                        continue
                    else:
                        cv2.destroyWindow('Frame viewer')
                        self.pedestrian_counter = self.action_analysis(frame_data, start_timestamp, end_timestamp,
                                                                       self.pedestrian_counter)

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

            if key == ord('q'):
                break

    def create_classes_for_intent_prediction(self):
        """
        Manual creation of classes for the dataset for the Action Recognition model.
        """
        frame_data = self.check_correct_frames_data()
        frame_data = frame_data.loc[frame_data['timestamp'] >= self.start_timestamp]
        color_frames = frame_data['rgb_frame'].tolist()
        depth_frames = frame_data['depth_frame'].tolist()
        timestamps = frame_data['timestamp'].tolist()

        recording = False
        start_timestamp = 0
        end_timestamp = 0

        index = 0
        while True:
            timestamp = timestamps[index]
            frame_path = color_frames[index]
            depth_path = depth_frames[index]
            frame_path = check_os_windows(frame_path)
            depth_path = check_os_windows(depth_path)

            frame = cv2.imread(frame_path)
            depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            added_images = self.combine_rgb_depth_frames(frame, depth_map, depth_threshold=10000)

            cv2.imshow('Frame viewer', added_images)

            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC key
                break

            if key == ord('r'):
                if not recording:
                    recording = True
                    start_timestamp = timestamps[index]
                    print(f'Start action recording at {start_timestamp}')
                    continue

                if recording:
                    recording = False
                    end_timestamp = timestamps[index]
                    print(f'Stop Recording at {end_timestamp}')

                    if start_timestamp >= end_timestamp:
                        print('End timestamp must be greater than start timestamp.')
                        continue
                    else:
                        cv2.destroyWindow('Frame viewer')
                        self.pedestrian_counter = self.action_analysis(frame_data, start_timestamp, end_timestamp,
                                                                       self.pedestrian_counter, check_intention=True)

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

            if key == ord('q'):
                break


    def action_analysis(self, frame_data, start_timestamp, end_timestamp, pedestrian_counter, check_intention=False):
        """
        review each pedestrian in the set of frames and labels their actions and final intent.
        Args:
            frame_data (pd.DataFrame): Dataframe with the Pedestrians data
            start_timestamp (str): Start timestamp
            end_timestamp (str): End timestamp
            pedestrian_counter (int): Counter for the pedestrians found in each section of the dataset
        """
        pedestrian_actions = {}
        pedestrian_intents = {}
        print(f"Checking pedestrian actions from {start_timestamp} to {end_timestamp}")

        frames_to_check = frame_data[(frame_data['timestamp'] >= start_timestamp) & (frame_data['timestamp'] <= end_timestamp)]
        checked_frames = 0
        pedestrian_intent = 0

        for _, row in frames_to_check.iterrows():
            checked_frames += 1
            print(f"Checking frame {checked_frames} of {len(frames_to_check) + 1}")
            pedestrians_in_frame = []
            color_frame_path = Path(row['rgb_frame'])
            depth_frame_path = Path(row['depth_frame'])
            timestamp = row['timestamp']

            color_frame_path = check_os_windows(color_frame_path)
            depth_frame_path = check_os_windows(depth_frame_path)

            frame = cv2.imread(color_frame_path)
            depth_map = cv2.imread(depth_frame_path, cv2.IMREAD_UNCHANGED)

            results = self.model(frame)
            persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]
            print("Pedestrians detected: ", len(persons))

            if len(persons) == 0:
                print("Zero persons detected ??????????????????")
                continue

            for person in persons:
                x1, y1, x2, y2, confidence, class_id = person.tolist()
                w = x2 - x1
                h = y2 - y1
                detection_data = [[int(x1), int(y1), int(w), int(h)], confidence, class_id]

                depth_values_of_roi = depth_map[int(y1):int(y2), int(x1):int(x2)]
                pedestrian_image = frame[int(y1):int(y2), int(x1):int(x2)]
                pedestrian_distance = estimate_pedestrian_distance(depth_values_of_roi)

                if confidence < 0.7:
                    print("Confidence too low")
                    cv2.imshow(f"Is this a pedestrian? confidence: {confidence}", pedestrian_image)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('n'):
                        cv2.destroyAllWindows()
                        continue
                    else:
                        cv2.destroyAllWindows()

                if pedestrian_distance > 10.0:
                    print("Pedestrian a bit far away")
                    cv2.imshow(f"Is pedestrian close enough? distance: {pedestrian_distance}" , pedestrian_image)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('n'):
                        cv2.destroyAllWindows()
                        continue
                    else:
                        cv2.destroyAllWindows()

                pedestrians_in_frame.append(detection_data)

            pedestrian_tracks = self.track_pedestrians(frame, np.array(pedestrians_in_frame, dtype="object"))

            for track in pedestrian_tracks:
                if not track.is_confirmed:
                    continue

                track_id = int(track.track_id) + pedestrian_counter
                ltrb = track.to_tlbr()
                x1, y1, x2, y2 = map(int, ltrb)
                pedestrian_image = frame[int(y1):int(y2), int(x1):int(x2)]
                pedestrian_distance = estimate_pedestrian_distance(depth_map[int(y1):int(y2), int(x1):int(x2)])
                pedestrian_position = estimate_pedestrian_position((x1, y1, x2, y2),
                                                                                        pedestrian_distance,
                                                                                        self.camera_intrinsics)

                print(f"Pedestrian estimated position is {pedestrian_position}")

                file_name = f"pedestrian_{track_id}.csv"

                if not check_file(file_name):
                    cv2.imshow(f"Pedestrian {track_id}", pedestrian_image)
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        break

                    pedestrian_action = input("Enter the pedestrian action: ")
                    pedestrian_actions[track_id] = pedestrian_action
                    pedestrian_intents[track_id] = 0

                    if check_intention:
                        pedestrian_intent = input(f"Does the pedestrian intent to change their current action ({pedestrian_action}? (0: No, 1: Yes)")
                        pedestrian_intents[track_id] = pedestrian_intent
                    cv2.destroyAllWindows()

                    pedestrian_data = {
                        'timestamp': timestamp,
                        'track_id': track_id,
                        'source': self.source,
                        'bounding_box': str([x1, y1, x2, y2]),
                        'confidence': confidence,
                        'distance': pedestrian_distance,
                        'position': pedestrian_position,
                        'action': pedestrian_actions[track_id],
                        'intent': pedestrian_intents[track_id]
                    }

                    # save the data as csv file

                    save_path = save_csv_file(pedestrian_data, file_name, self.output_folder, create=True)

                    cv2.imwrite(save_path + f"pedestrian_{track_id}_{timestamp}.png", pedestrian_image)


                # if the file already exists, open the csv file and append the new data
                else:
                    pedestrian_data = {
                        'timestamp': timestamp,
                        'track_id': track_id,
                        'source': self.source,
                        'bounding_box': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'distance': pedestrian_distance,
                        'position': pedestrian_position,
                        'action': pedestrian_actions[track_id],
                        'intent': pedestrian_intents[track_id]
                    }

                    # save the data as csv file
                    save_path = save_csv_file(pedestrian_data, file_name, self.output_folder, create=False)
                    cv2.imwrite(save_path + f"pedestrian_{track_id}_{timestamp}.png", pedestrian_image)

        return (pedestrian_counter +len(pedestrian_tracks))


    def track_pedestrians(self, frame, detections):
        """Track pedestrians using DeepSort tracker
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        return tracks


    def combine_rgb_depth_frames(self, frame, depth_map, depth_threshold=10000):
        """
        Combine a RGB and a Depth Map into a single image.
        """
        # if values from the depth map are greater than 10000, make them 0
        depth_map[depth_map > depth_threshold] = 0

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_HOT)
        combined = cv2.addWeighted(frame, 0.7, depth_colormap, 0.3, 0)

        return combined

    def save_frames(self, start_timestamp, end_timestamp, action, intent, pedestrian_counter):
        """
        Save the frames from the start timestamp to the end timestamp in a folder.

        Args:
            start_timestamp (str): Start timestamp
            end_timestamp (str): End timestamp
            action (str): Action to save the frames
        """
        print(f"Saving frames from {start_timestamp} to {end_timestamp} for action {action} number {pedestrian_counter}")

        if self.check_intention:
            if intent == 0:
                class_name = f"{action}_no_intent"
            elif intent == 1:
                class_name = f"{action}_with_intent"
            action_folder = self.output_folder / class_name / f"{action}_{pedestrian_counter}"
        else:
            action_folder = self.output_folder / f"{action}_{pedestrian_counter}"

        action_folder.mkdir(parents=True, exist_ok=True)

        frame_data = self.check_correct_frames_data()
        frames_to_save = frame_data[(frame_data['timestamp'] >= start_timestamp) & (frame_data['timestamp'] <= end_timestamp)].copy()

        # Explicitly cast the 'action' column to string dtype
        frames_to_save.loc['action'] = frames_to_save['action'].astype  (str)
        frames_to_save.loc[:, 'action'] = action # or any default value

        if 'intent' not in frames_to_save.columns:
            frames_to_save.loc[:,'intent'] = 0 # or any default value

        # Explicitly cast the 'intent' column to integer dtype
        frames_to_save.loc['intent'] = frames_to_save['intent'].astype(int)
        frames_to_save.loc[:, 'intent'] = intent

        csv_output_path = action_folder / f"{action}_{pedestrian_counter}.csv"
        frames_to_save.to_csv(csv_output_path, index=False)

        for _, row in frames_to_save.iterrows():
            color_frame_path = check_os_windows(row['rgb_frame'])
            depth_frame_path = check_os_windows(row['depth_frame'])
            if color_frame_path.exists() and depth_frame_path.exists():
                shutil.copy(color_frame_path, action_folder / color_frame_path.name)
                shutil.copy(depth_frame_path, action_folder / depth_frame_path.name)
            else:
                print(f"Frame {row['rgb_frame']} does not exist.")



def main():
    data_path = '/media/felipezero/T7 Shield/DATA/thesis/Videos/video_01/'
    output_folder = '/media/felipezero/T7 Shield/DATA/thesis/intent_prediction_dataset/classes_01/'

    dataset_manager = DatasetManager(data_path=data_path, output_folder=output_folder)
    dataset_manager.create_classes_for_intent_prediction()

if __name__ == '__main__':
    main()