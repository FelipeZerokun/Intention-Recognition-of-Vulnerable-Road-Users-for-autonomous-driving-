
import torch

from project_utils.project_utils import estimate_pedestrian_distance


class PedestrianDetector:

    def __init__(self, confidence_threshold: float = 0.50, distance_threshold: float = 30.0):
        """
        Class for pedestrian detection using YOLOv5.
        The confidence threshold will prevent low confidence detections.
        The distance threshold will prevent tracking pedestrians that are too far away from the robot.
        Args:
            confidence_threshold: Min Confidence threshold for pedestrian detection.
            distance_threshold: Max Distance threshold for pedestrian tracking.
        """

        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold


    def detect_pedestrians(self, color_frame, depth_frame):
        """
        Detect pedestrians in the frame using YOLOv5 model.
        """
        detection_results = []
        results = self.model(color_frame)
        persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]

        if len(persons) == 0:
            return None

        for person in persons:
            x1, y1, x2, y2, confidence, class_id = person.tolist()
            w = x2 - x1
            h = y2 - y1
            detection_data = [[int(x1), int(y1), int(w), int(h)], confidence, class_id]

            depth_values_of_roi = depth_frame[int(y1):int(y2), int(x1):int(x2)]
            pedestrian_distance = estimate_pedestrian_distance(depth_values_of_roi)

            if pedestrian_distance > self.distance_threshold or confidence < self.confidence_threshold:
                continue

            detection_results.append(detection_data)

        return detection_results

