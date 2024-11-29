from pathlib import Path
import os
import shutil

import cv2
import pandas as pd
import torch
import numpy as np

from deep_sort_realtime.deepsort_tracker import DeepSort


from project_utils import check_path, check_file, check_os_windows, estimate_pedestrian_distance


class PedestrianTracker:

    def __init__(self, frames_data: Path):
        """
        Class to create a dataset with human actions from several frames extracted from a video file.

        Args:
            video_data (Path): Path to the CSV file with the frame's data.

        """
        self._DATA_COLUMNS = ['timestamp', 'Robot odometry', 'rgb_frame', 'depth_frame']
        self.frames_data = check_os_windows(frames_data)
        check_file(self.frames_data)

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.detection_results = []

        self.tracker = DeepSort(
            max_age=30,
            nn_budget=70,
            nms_max_overlap=1.0,
        )

        self.walking_counter = 0
        self.standing_still_counter = 0
        self.pedestrian_counter = 0


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

    def track_pedestrian(self, pedestrian_number: int):
        frame_data = self.check_correct_frames_data()
        color_frames = frame_data['rgb_frame'].tolist()
        depth_frames = frame_data['depth_frame'].tolist()
        timestamps = frame_data['timestamp'].tolist()
        x_pos = frame_data['Robot odometry'].tolist()

        pedestrians_in_frame = []

        for i in range(len(color_frames)):

            color_frame = cv2.imread(color_frames[i])
            depth_map = cv2.imread(depth_frames[i], cv2.IMREAD_UNCHANGED)
            timestamp = timestamps[i]

            results = self.model(color_frame)
            persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]

            if len(persons) == 0:
                continue

            for person in persons:
                pedestrians_in_frame = []
                x1, y1, x2, y2, confidence, class_id = person.tolist()
                w = x2 - x1
                h = y2 - y1
                detection_data = [[int(x1), int(y1), int(w), int(h)], confidence, class_id]

                depth_values_of_roi = depth_map[int(y1):int(y2), int(x1):int(x2)]
                pedestrian_distance = estimate_pedestrian_distance(depth_values_of_roi)

                if pedestrian_distance > 10.0 or confidence < 0.7:
                    continue

                pedestrians_in_frame.append(detection_data)

            if len(pedestrians_in_frame) == 0:
                continue

            pedestrian_tracks = self.track_pedestrians(color_frame, np.array(pedestrians_in_frame, dtype="object"))

            for track in pedestrian_tracks:
                if not track.is_confirmed:
                    continue

                track_id = track.track_id
                ltrb = track.to_tlbr() # Get the bounding boxqqqq
                x1, y1, x2, y2 = map(int, ltrb)

                depth_values_of_roi = depth_map[int(y1):int(y2), int(x1):int(x2)]
                pedestrian_image = color_frame[int(y1):int(y2), int(x1):int(x2)]
                pedestrian_distance = estimate_pedestrian_distance(depth_values_of_roi)

                print(f'Track ID: {track_id}')
                print(f'Depth: {pedestrian_distance:.2f} m')

                # Draw the bounding box
                cv2.rectangle(color_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(color_frame, f'Track ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
                cv2.putText(color_frame, f'Depth: {pedestrian_distance:.2f} m', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

            cv2.imshow('Pedestrian detection', color_frame)
            if cv2.waitKey(500) & 0xFF == ord('q'):
                break



    def track_pedestrians(self, frame, detections):
        """Track pedestrians using DeepSort tracker
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        return tracks


def main():
    frames_data = '/media/felipezero/T7 Shield/DATA/thesis/Videos/frames/navigation_data.csv'

    pedestrian_tracker = PedestrianTracker(frames_data)
    pedestrian_tracker.track_pedestrian(1)



if __name__ == '__main__':
    main()
