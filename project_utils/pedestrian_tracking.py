from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms

from project_utils.project_utils import estimate_pedestrian_distance


class PedestrianTracker:

    """
    Class that will track pedestrians through frames.
    """

    def __init__(self, model, max_age: int = 5, nn_budget: int = 70, nms_max_overlap: float = 1.0):
        """
        Args:
            max_age: Maximum number of frames that a track is not updated.
            nn_budget: Maximum number of features to keep in memory.
            nms_max_overlap: Non-maximum suppression threshold.
            sequence_lenght: Number of frames to consider for action recognition.
            device: Device to run the model on.
        """
        self.model = model

        self.sequence_length = model.sequence_length
        self.device = model.device
        self.frame_buffers = {}
        self.tracker = DeepSort(max_age=max_age,
                                nn_budget=nn_budget,
                                nms_max_overlap=nms_max_overlap
        )

        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])


    def track_pedestrians(self, frame, depth_frame, pedestrians_in_frame):

        pedestrian_tracks = self.update_tracker(frame, np.array(pedestrians_in_frame, dtype="object"))

        for track in pedestrian_tracks:
            if not track.is_confirmed:
                continue

            track_id = int(track.track_id) # unique track id
            ltrb = track.to_tlbr()
            x1, y1, x2, y2  = map(int, ltrb)

            # Estimate the distance to the pedestrian
            pedestrian_depth_roi = depth_frame[y1:y2, x1:x2]
            pedestrian_distance = estimate_pedestrian_distance(pedestrian_depth_roi)

            pedestrian_tensor = self.crop_and_process_pedestrian(frame, (x1, y1, x2, y2))

            ## Initialize the buffer for the track_id if it does not exist
            if track_id not in self.frame_buffers:
                self.frame_buffers[track_id] = []

            # Add the cropped tensor to the buffer
            self.frame_buffers[track_id].append(pedestrian_tensor)

            if len(self.frame_buffers[track_id]) > self.sequence_length:
                # Stack the sequence into a tensor
                sequence = torch.stack(self.frame_buffers[track_id][-self.sequence_length:], dim=2)  # Shape: (C, T, H, W)
                sequence = sequence.unsqueeze(0)  # Add batch dimension: (B, C, T, H, W)

                # Fix the shape by removing the extra dimension
                sequence = sequence.squeeze(1)  # Shape: (B, C, T, H, W)

                # Predict the action
                action, probability = self.model.action_prediction(sequence)

                # Draw the bounding box
                frame = self.draw_bbox_in_frame(frame, (x1, y1, x2, y2), track_id, action, probability, pedestrian_distance)

        return frame

    def update_tracker(self, frame, detections):
        """Track pedestrians using DeepSort tracker
        Args:
            frame (np.ndarray): Input frame.
            detections (np.ndarray): Detected bounding boxes.
        Returns:
            list: List of updated tracks.
            """
        tracks = self.tracker.update_tracks(detections, frame=frame)

        return tracks

    def crop_and_process_pedestrian(self, frame, bbox):
        """
        Crop the pedestrian bounding box from the frame and apply transforms.

        Args:
            frame (np.ndarray): Input frame.
            pedestrian (list): Bounding box coordinates [x, y, w, h].

        Returns:
            torch.Tensor: Transformed pedestrian image.
        """

        x1, y1, x2, y2 = bbox
        pedestrian_img = frame[int(y1):int(y2), int(x1):int(x2)]

        # Convert to PIL Image
        pedestrian_img = Image.fromarray(pedestrian_img)
        cropped_tensor = self.transform(pedestrian_img).unsqueeze(0)
        return cropped_tensor.to(self.device)
    def draw_bbox_in_frame(self, frame, bbox, id, action, probability, distance):
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
        cv2.putText(frame, f'{id}', (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        cv2.putText(frame, f'{distance:.2f} m', (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        if action == 'Walking':
            cv2.putText(frame, f'{action}', (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(frame, f'{probability:.3f}', (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        elif action == 'Standing Still':
            cv2.putText(frame, f'{action}', (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, f'{probability:.3f}', (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return frame
