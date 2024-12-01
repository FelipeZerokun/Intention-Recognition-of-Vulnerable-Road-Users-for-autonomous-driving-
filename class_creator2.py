from pathlib import Path
import os
import shutil

import cv2
import pandas as pd
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

from project_utils import check_path, check_file, check_os_windows, estimate_pedestrian_distance


class HumanActionClassCreation:
    def __init__(self, frames_data: Path, output_folder: Path):
        """
        Class to create a dataset with human actions from several frames extracted from a video file.

        Args:
            video_data (Path): Path to the CSV file with the frame's data.

        """
        self._DATA_COLUMNS = ['timestamp', 'Robot odometry', 'rgb_frame', 'depth_frame']
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        self.tracker = DeepSort(
            max_age=30,
            nn_budget=70,
            nms_max_overlap=1.0,
        )

        self.frames_data = check_os_windows(frames_data)

        if not check_file(self.frames_data):
            raise FileNotFoundError(f'File {self.frames_data} not found.')

        self.output_walking_folder = output_folder + 'walking/'
        self.output_standing_still_folder = output_folder + 'standing_still/'
        check_path(folder_path=self.output_walking_folder, create=True)
        check_path(folder_path=self.output_standing_still_folder, create=True)

        self.walking_counter = 0
        self.standing_still_counter = 0
        self.pedestrian_counter = 15

        self.create_classes()

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

    def create_classes(self):
        """
        Manual creation of classes for human actions in the frames.
        """
        frame_data = self.check_correct_frames_data()
        color_frames = frame_data['rgb_frame'].tolist()
        depth_frames = frame_data['depth_frame'].tolist()
        timestamps = frame_data['timestamp'].tolist()

        recording = False
        start_timestamp = 0
        end_timestamp = 0

        index = 0
        while True:
            frame_path = color_frames[index]
            depth_path = depth_frames[index]
            frame_path = check_os_windows(frame_path)
            depth_path = check_os_windows(depth_path)

            frame = cv2.imread(frame_path)
            depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            # if values from the depth map are greater than 5000, make them 0
            depth_map[depth_map > 6000] = 0

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_HOT)
            added_images = cv2.addWeighted(frame, 0.7, depth_colormap, 0.3, 0)

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
                        self.pedestrian_counter = self.action_analysis(frame_data, start_timestamp, end_timestamp, self.pedestrian_counter)

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

        cv2.destroyAllWindows()

    def action_analysis(self, frame_data, start_timestamp, end_timestamp, pedestrian_counter):
        """
        review each pedestrian in the set of frames and labels their actions
        Args:
            frame_data (pd.DataFrame): Dataframe with the Pedestrians data
            start_timestamp (str): Start timestamp
            end_timestamp (str): End timestamp
            pedestrian_counter (int): Counter for the pedestrians found in each section of the dataset
        """
        pedestrian_actions = {}
        print(f"Checking pedestrian actions from {start_timestamp} to {end_timestamp}")

        frames_to_check = frame_data[(frame_data['timestamp'] >= start_timestamp) & (frame_data['timestamp'] <= end_timestamp)]
        checked_frames = 0

        for _, row in frames_to_check.iterrows():
            checked_frames += 1
            print(f"Checking frame {checked_frames} of {len(frames_to_check) + 1}")
            pedestrians_in_frame = []
            color_frame_path = Path(row['rgb_frame'])
            depth_frame_path = Path(row['depth_frame'])
            timestamp = row['timestamp']
            odometry = row['Robot odometry']

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
                pedestrian_distance = estimate_pedestrian_distance(depth_values_of_roi)
                pedestrian_image = frame[int(y1):int(y2), int(x1):int(x2)]

                if confidence < 0.7:
                    print("Confidence too low")
                    cv2.imshow("Is this a pedestrian?", pedestrian_image)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord('n'):
                        cv2.destroyAllWindows()
                        continue
                    else:
                        cv2.destroyAllWindows()

                if pedestrian_distance > 10.0:
                    print("Pedestrian a bit far away")
                    cv2.imshow("Is pedestrian close enough?", pedestrian_image)
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

                file_name = f"pedestrian_{track_id}.csv"

                if not check_file(file_name):
                    cv2.imshow(f"Pedestrian {track_id}", pedestrian_image)
                    if cv2.waitKey(500) & 0xFF == ord('q'):
                        break
                    pedestrian_action = input("Enter the pedestrian action: ")
                    pedestrian_actions[track_id] = pedestrian_action
                    cv2.destroyAllWindows()

                    pedestrian_data = {
                        'timestamp': timestamp,
                        'odometry': odometry,
                        'velocity': row['Robot estimated velocity'],
                        'bounding_box': str([x1, y1, x2, y2]),
                        'confidence': confidence,
                        'distance': pedestrian_distance,
                        'action': pedestrian_actions[track_id]
                    }

                    # save the data as csv file
                    if pedestrian_action == 'walking':
                        pedestrian_data_dir = self.output_walking_folder + f'pedestrian_{track_id}/'
                        pedestrian_data_file = pedestrian_data_dir + file_name

                    elif pedestrian_action == 'standing_still':
                        pedestrian_data_dir = self.output_standing_still_folder + f'pedestrian_{track_id}/'
                        pedestrian_data_file = pedestrian_data_dir + file_name


                    check_path(folder_path=pedestrian_data_dir, create=True)
                    pedestrian_data_df = pd.DataFrame([pedestrian_data])
                    pedestrian_data_df.to_csv(pedestrian_data_file, index=False)
                    pedestrian_data_df.to_csv(file_name, index=False)
                    cv2.imwrite(pedestrian_data_dir + f"pedestrian_{track_id}_{timestamp}.png", pedestrian_image)


                # if the file already exists, open the csv file and append the new data
                else:
                    pedestrian_data = {
                        'timestamp': timestamp,
                        'odometry': odometry,
                        'velocity': row['Robot estimated velocity'],
                        'bounding_box': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'distance': pedestrian_distance,
                        'action': pedestrian_actions[track_id]
                    }

                    # save the data as csv file
                    if pedestrian_actions[track_id] == 'walking':
                        pedestrian_data_dir = self.output_walking_folder + f'pedestrian_{track_id}/'
                        pedestrian_data_file = pedestrian_data_dir + file_name

                    elif pedestrian_actions[track_id] == 'standing_still':
                        pedestrian_data_dir = self.output_standing_still_folder + f'pedestrian_{track_id}/'
                        pedestrian_data_file = pedestrian_data_dir + file_name

                    pedestrian_data_df = pd.read_csv(pedestrian_data_file)
                    pedestrian_data_df = pd.concat([pedestrian_data_df, pd.DataFrame([pedestrian_data])], ignore_index=True)
                    pedestrian_data_df.to_csv(pedestrian_data_file, index=False)
                    pedestrian_data_df.to_csv(file_name, index=False)
                    cv2.imwrite(pedestrian_data_dir + f"pedestrian_{track_id}_{timestamp}.png", pedestrian_image)

        return (pedestrian_counter +len(pedestrian_tracks))


    def track_pedestrians(self, frame, detections):
        """Track pedestrians using DeepSort tracker
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        return tracks

def main():
    frames_data = '/media/felipezero/T7 Shield/DATA/thesis/Videos/video_01/navigation_data.csv'
    output_folder = '/media/felipezero/T7 Shield/DATA/thesis/Videos/video_01/classes/'

    HumanActionClassCreation(frames_data=frames_data, output_folder=output_folder)


if __name__ == '__main__':
    main()