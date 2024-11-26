import torch
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from project_utils import (check_path, check_file, estimate_pedestrian_distance)
import os

from deep_sort_realtime.deepsort_tracker import DeepSort

#load YOLOv5 model
class PedestrianRecognition:

    _DEPTH_UNIT_SCALE = 0.001
    def __init__(self, data_directory: Path):
        """Class for pedestrian recognition using YOLOv5 model
        """
        self.classes_dir = data_directory
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.detection_results = []

        self.tracker = DeepSort(
            max_age = 30,
            nn_budget=70,
            nms_max_overlap=1.0,
        )

    def get_class(self):
        """ Checks if the directory has several floders representing the classes and
        if there is a csv file with the data of each frame"""

        detected_class = self.classes_dir.parts[-1]
        check_path(folder_path=self.classes_dir, create=False)

        print("Class found: ", detected_class)
        class_data = Path(self.classes_dir, f"{detected_class}.csv")
        check_file(class_data)
        class_data = pd.read_csv(class_data)

        return class_data



    def pedestrian_detect_and_track(self):
        """Apply YOLOv5 model to detect pedestrians in the frames
        """

        frame_data = self.get_class()

        for index, row in frame_data.iterrows():
            timestamp = row['timestamp']
            odometry = row['Robot odometry']
            estimated_vel = row['Robot estimated velocity']
            color_frame_path = row['rgb_frame']
            depth_frame_path = row['depth_frame']
            min_depth = 0.3
            max_depth = 20.0

            pedestrians_in_frame = []
            
            if os.name == 'nt':
                color_frame_path = color_frame_path.replace('/media/felipezero/T7 Shield/', 'D:/')
                depth_frame_path = depth_frame_path.replace('/media/felipezero/T7 Shield/', 'D:/')

            color_frame = cv2.imread(color_frame_path)
            depth_map = cv2.imread(depth_frame_path, cv2.IMREAD_UNCHANGED)
            depth_frame = cv2.imread(depth_frame_path, cv2.IMREAD_GRAYSCALE)

            results = self.model(color_frame)
            persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]

            if len(persons) == 0:
                cv2.imshow('Pedestrian detection', color_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    continue

            for person in persons:
                x1, y1, x2, y2, confidence, class_id = person.tolist()
                w = x2 - x1
                h = y2 - y1
                detection_data = [[int(x1), int(y1), int(w), int(h)], confidence, class_id]

                depth_values_of_roi = depth_map[int(y1):int(y2), int(x1):int(x2)]
                pedestrian_distance = estimate_pedestrian_distance(depth_values_of_roi)

                detection = {
                    'timestamp': timestamp,
                    'odometry': odometry,
                    'velocity': estimated_vel,
                    'bounding_box': [x1, y1, x2, y2],
                    'confidence': confidence
                }

                if confidence < 0.5 or pedestrian_distance > max_depth or pedestrian_distance < min_depth:
                    continue

                self.detection_results.append(detection)
                pedestrians_in_frame.append(detection_data)

            pedestrian_tracks = self.track_pedestrians(color_frame, np.array(pedestrians_in_frame, dtype="object"))

            for track in pedestrian_tracks:
                if not track.is_confirmed:
                    continue

                track_id = track.track_id
                ltrb = track.to_tlbr() # Get the bounding box
                x1, y1, x2, y2 = map(int, ltrb)

                # Draw the bounding box
                cv2.rectangle(color_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(color_frame, f'Track ID: {track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(color_frame, f'Depth: {pedestrian_distance:.2f} m', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow('Pedestrian detection', color_frame)
            if cv2.waitKey(5000) & 0xFF == ord('q'):
                break




            # for person in persons:
            #     x1, y1, x2, y2, confidence, class_id = person
            #     if confidence < 0.5:
            #         continue
            #     cv2.rectangle(color_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            #     cv2.rectangle(depth_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            #     roi_depth_values = depth_map[int(y1):int(y2), int(x1):int(x2)]
            #     pedestrian_distance = estimate_pedestrian_distance(roi_depth_values)
            #
            #     cv2.putText(color_frame, f'Person {confidence:.2f}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #     cv2.putText(color_frame, f'Depth: {pedestrian_distance:.2f} m', (int(x1), int(y1 - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_HOT)
            # added_images = cv2.addWeighted(color_frame, 0.7, depth_colormap, 0.3, 0)
            # merged_images = cv2.hconcat([added_images, cv2.cvtColor(depth_frame, cv2.COLOR_GRAY2RGB)])
            #
            # cv2.imshow('Pedestrian detection', merged_images)
            # if cv2.waitKey(2000) & 0xFF == ord('q'):
            #     break

        cv2.destroyAllWindows()


    def detect_pedestrians(self, frame):
        """Detect pedestrians in the frame
        """
        results = self.model(frame)
        persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]
        return persons

    def track_pedestrians(self, frame, detections):
        """Track pedestrians using DeepSort tracker
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        return tracks

    def bbox_draw(self, frame, bbox, color=(0, 255, 0), thickness=2):
        """Draw bounding box on the frame
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        return frame



def main():

    frames_path = Path('/media/felipezero/T7 Shield/DATA/thesis/Videos/frames/classes/')
    # frames_path = Path('D:/DATA/thesis/Videos/frames/classes/')
    class_01 = Path(frames_path / 'walking_3')

    pedestrian_recognition = PedestrianRecognition(data_directory=class_01)

    pedestrian_recognition.pedestrian_detect_and_track()

if __name__ == '__main__':
    main()
