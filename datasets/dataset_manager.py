from pathlib import Path
import shutil
from typing import Tuple, List, Dict, Any

import cv2
import numpy as np
import pandas as pd
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

from project_utils.project_utils import (
    check_path,
    check_file,
    check_os_windows,
    get_data,
    estimate_pedestrian_distance,
    estimate_pedestrian_position,
    save_csv_file
)


class DatasetManager:
    """
    A class to manage datasets for human action and intent recognition.
    It uses data extracted from RGB and depth frames, along with YOLOv5 for detection.


    This class handles loading and validating sensor and image data, tracking pedestrians using
    YOLOv5 and DeepSORT, and providing interactive tools for manually labeling pedestrian actions
    and intentions. It supports creating datasets for both Action Recognition and Intent Prediction
    models by combining RGB and depth information and saving pedestrian-specific data sequences.

    Attributes:
        data_path (str): Path to the dataset directory.
        source (str): Name of the video or source folder.
        output_folder (str): Path to the output folder for saving processed data.
        frames_data (Path): Path to the navigation data CSV file.
        camera_info (Path): Path to the camera information CSV file.
        image_size (List[int]): Image dimensions [height, width].
        camera_intrinsics (np.ndarray): Intrinsic camera matrix.
        model (torch.nn.Module): Pretrained YOLOv5 model for pedestrian detection.
        tracker (DeepSort): DeepSort tracker for pedestrian tracking.
        pedestrian_counter (int): Counter for pedestrian identification.
        start_timestamp (int): Initial timestamp for filtering frames.

    Methods:
        get_camera_info():
            Loads camera parameters including image size and intrinsic matrix.

        check_correct_frames_data():
            Validates that all required columns exist in the navigation CSV file.

        create_classes_for_action_recognition():
            Interactive method to annotate sequences for action recognition, allowing the user
            to view frame pairs and define action segments.

        create_classes_for_intent_prediction():
            Similar to action class creation, but prompts intent labeling for each pedestrian
            based on visual and distance cues.

        action_analysis():
            Processes all pedestrian detections within a timestamp window to confirm pedestrian
            presence, estimate distance and position, and prompt the user to annotate their
            actions and intentions.
    """
    def __init__(self, data_path: str, video, output_folder: str):
        """
        Initializes the DatasetManager by setting paths, loading YOLOv5 and DeepSORT models,
        and preparing camera info data.
        Args:
            data_path (str): Path to the dataset directory.
            video (str): Name of the video or source folder.
            output_folder (str): Path to the output folder for saving processed data.
        """
        self._DATA_COLUMNS = ['timestamp', 'Robot odometry', 'rgb_frame', 'depth_frame']

        self.frames_data = Path(data_path,video, 'navigation_data.csv')
        self.camera_info = Path(data_path, video, 'camera_info.csv')

        self.data_path = data_path
        self.source = video

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

        self.pedestrian_counter = 9
        self.start_timestamp = 1683276896237854159

    def get_camera_info(self) -> Tuple[List[int], np.ndarray]:
        """
        Loads camera parameters from a CSV file, including image size and intrinsic camera matrix.
        Returns:
            Tuple[List[int], np.ndarray]: A tuple containing the image size as a list [height, width]
            and the intrinsic camera matrix as a numpy array.

        """

        camera_info = pd.read_csv(self.camera_info)
        image_height = camera_info.loc[0, 'image_height']
        image_width = camera_info.loc[0, 'image_width']

        intrinsic_matrix_str = camera_info.loc[0, 'Intrinsic_camera_matrix']

        # Convert the string representation of the matrix to a numpy array
        intrinsic_matrix = np.fromstring(intrinsic_matrix_str.strip('[]'), sep=' ').reshape((3, 3))

        return ([image_height, image_width], intrinsic_matrix)

    def check_correct_frames_data(self) -> pd.DataFrame:
        """
        Validates that the frames data CSV file contains all required columns for processing.
        Returns:
            pd.DataFrame: The DataFrame containing the frames data if validation is successful.
        Raises:
            ValueError: If any required column is missing from the frames data CSV file.

        """

        frame_data = pd.read_csv(self.frames_data)

        columns = frame_data.columns

        for column in self._DATA_COLUMNS:
            if column not in columns:
                raise ValueError(f'Column {column} not found in the frames data CSV file.')

        return frame_data

    def create_classes_for_action_recognition(self) -> None:
        """
        Launches a manual annotation interface to label frame sequences for pedestrian action.

        The user can label actions for pedestrians detected in the frames, and the sequences are saved
        in the specified output folder.
            - Use 'a' to move to the previous frame.
            - Use 'd' to move to the next frame.
            - Use 'r' to start/stop recording a sequence of frames.
            - The frames between the start and end timestamps will be analyzed for pedestrians action.
            - End timestamp must be greater than the start timestamp.

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
            frame_path = check_os_windows(self.data_path + frame_path)
            depth_path = check_os_windows(self.data_path + depth_path)

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

    def create_classes_for_intent_prediction(self) -> None:
        """
        Launches a manual annotation interface to label frame sequences for pedestrian intent.

        The user can label actions for pedestrians detected in the frames, and the sequences are saved
        in the specified output folder.
            - Use 'a' to move to the previous frame.
            - Use 'd' to move to the next frame.
            - Use 'r' to start/stop recording a sequence of frames.
            - The frames between the start and end timestamps will be analyzed for pedestrians action AND intention.
            - End timestamp must be greater than the start timestamp.

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
            frame_path = check_os_windows(self.data_path + frame_path)
            depth_path = check_os_windows(self.data_path + depth_path)

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

    def action_analysis(
        self,
        frame_data: pd.DataFrame,
        start_timestamp: str,
        end_timestamp: str,
        pedestrian_counter: int,
        check_intention: bool = False
    ) -> int:
        """
        Analyzes pedestrian behavior within a specific time window and stores
        pedestrian labeled data based on manual input and automated tracking.

        Pedestrians are detected using YOLOv5, tracked using DeepSORT, and filtered
        based on detection confidence and distance from the robot. The user confirms
        each detection and assigns a class label via terminal input.

        Args:
            frame_data (pd.DataFrame): Frame metadata including timestamps and paths.
            start_timestamp (str): Timestamp marking the start of the labeling segment.
            end_timestamp (str): Timestamp marking the end of the labeling segment.
            pedestrian_counter (int): Counter to index unique pedestrians.
            check_intention (bool, optional): Whether to collect intention labels. Defaults to False.

        Returns:
            int: Updated pedestrian counter after processing the segment.
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
            color_frame_path = row['rgb_frame']
            depth_frame_path = row['depth_frame']
            timestamp = row['timestamp']
            ned_coordinates = row['Robot NED position']
            robot_velocity = row['Robot estimated velocity']

            robot_position = get_data(data=ned_coordinates, mode='NED')
            robot_velocity = get_data(data=robot_velocity, mode='velocity')


            color_frame_path = check_os_windows(self.data_path + color_frame_path)
            depth_frame_path = check_os_windows(self.data_path + depth_frame_path)

            frame = cv2.imread(color_frame_path)
            depth_map = cv2.imread(depth_frame_path, cv2.IMREAD_UNCHANGED)

            results = self.model(frame)
            persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]
            print("Pedestrians detected: ", len(persons))

            if len(persons) == 0:
                print("Zero persons detected!")
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
                        'robot_position': robot_position,
                        'robot_velocity': robot_velocity,
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
                        'robot_position': robot_position,
                        'robot_velocity': robot_velocity,
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


    def track_pedestrians(self, frame: np.ndarray, detections: np.ndarray) -> List[Any]:
        """
        Tracks pedestrians using the DeepSort tracker.

        Args:
            frame (np.ndarray): The current video frame.
            detections (np.ndarray): Array of detections in the current frame.

        Returns:
            List[Any]: List of tracked pedestrian objects.
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        return tracks


    def combine_rgb_depth_frames(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        depth_threshold: int = 10000
    ) -> np.ndarray:
        """
        Combines an RGB frame and a depth map into a single image for visualization.

        Args:
            frame (np.ndarray): The RGB image.
            depth_map (np.ndarray): The depth map corresponding to the RGB image.
            depth_threshold (int, optional): Threshold to cap depth values. Defaults to 10000.

        Returns:
            np.ndarray: Combined image with depth overlay.
        """
        # if values from the depth map are greater than 10000, make them 0
        depth_map[depth_map > depth_threshold] = 0

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_HOT)
        combined = cv2.addWeighted(frame, 0.7, depth_colormap, 0.3, 0)

        return combined

    def save_frames(
            self,
            start_timestamp: str,
            end_timestamp: str,
            action: str,
            intent: int,
            pedestrian_counter: int
    ) -> None:
        """
        Saves frames and metadata for a specific action and intent.

        Args:
            start_timestamp (str): Start timestamp for the frames to save.
            end_timestamp (str): End timestamp for the frames to save.
            action (str): Action label for the pedestrian.
            intent (int): Intent label (0: no intent, 1: with intent).
            pedestrian_counter (int): Counter for pedestrian identification.
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
    data_path = '/media/felipezero/T7 Shield/DATA/thesis/Videos/2023_05_05/'
    video = 'video_02'
    output_folder = '/media/felipezero/T7 Shield/DATA/thesis/intent_prediction_dataset/classes_02/'

    dataset_manager = DatasetManager(data_path=data_path, video=video, output_folder=output_folder)
    dataset_manager.create_classes_for_intent_prediction()

if __name__ == '__main__':
    main()