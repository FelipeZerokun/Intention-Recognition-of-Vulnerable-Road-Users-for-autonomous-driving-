from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import cv2

from utils.project_utils import estimate_pedestrian_distance


class PedestrianTracker:

    """
    Class that will track pedestrians through frames.
    """

    def __init__(self, max_age: int = 10, nn_budget: int = 70, nms_max_overlap: float = 1.0):
        """
        Args:
            max_age: Maximum number of frames that a track is not updated.
            nn_budget: Maximum number of features to keep in memory.
            nms_max_overlap: Non-maximum suppression threshold.
        """
        self.tracker = DeepSort(max_age=max_age, nn_budget=nn_budget, nms_max_overlap=nms_max_overlap)



    def track_pedestrians(self, frame, depth_frame, pedestrians_in_frame):
        """
        Track pedestrians in the frame.
        Args:
            frame:
            pedestrians:

        Returns:

        """
        pedestrian_tracks = self.track_pedestrians(frame, np.array(pedestrians_in_frame, dtype="object"))

        for track in pedestrian_tracks:
            if not track.is_confirmed:
                continue

            track_id = int(track.track_id) # unique track id
            ltrb = track.to_tlbr()
            x1, y1, x2, y2 = map(int, ltrb)
            pedestrian_image = frame[int(y1):int(y2), int(x1):int(x2)]
            pedestrian_distance = estimate_pedestrian_distance(depth_frame[int(y1):int(y2), int(x1):int(x2)])
            distance_to_object = track.get_distance()
            pedestrian_speed = track.get_speed()

            # Draw the bounding box
            self.draw_bounding_box(frame, (x1, y1, x2, y2), track_id, pedestrian_distance)


    def draw_bbox_in_frame(self, frame, bbox, id, distance):
        """
        Draw bounding box in the frame with the ID and distance of the pedestrian.
        Args:
             frame (np.array): Frame to draw the bounding box.
                bbox (tuple): Bounding box coordinates (x1, y1, x2, y2).
                id (int): ID of the pedestrian.
                distance (float): Estimated distance to the pedestrian.
        """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f'Track ID: {id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        cv2.putText(frame, f'Depth: {distance:.2f} m', (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)
        cv2.imshow('Pedestrian detection', frame)
        if cv2.waitKey(500) & 0xFF == ord('q'):
            return