import os
import cv2
import torch

from project_utils.pedestrian_tracking import PedestrianTracker
from project_utils.pedestrian_detection import PedestrianDetector
from project_utils.project_utils import check_path, check_file
from my_models.action_recognition_model import ActionRecognitionModel



def main():
    video_dir = "D:/DATA/thesis/Videos/video_04"
    check_path(video_dir)
    model_dir = "../results/action_recognition_model.pth"
    if not check_file(model_dir):
        print(f"Model file not found: {model_dir}")
        raise FileNotFoundError

    # Action label mapping (customize this based on your dataset)
    ACTION_LABELS = {0: "Standing Still", 1: "Walking"}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pre-trained action recognition model
    my_model = ActionRecognitionModel(action_labels=ACTION_LABELS, model_dir=model_dir, device=device)

    my_model.model.eval().to(device)

    detector = PedestrianDetector(confidence_threshold=0.6,
                                  distance_threshold=22.0
    )

    tracker = PedestrianTracker(model=my_model,
                                max_age=4,
                                nn_budget=70,
                                nms_max_overlap=1.0,
    )

    depth_timestamp = None
    color_timestamp = None

    for frame in os.listdir(video_dir):
        frame_type = frame.split('_')[1]
        timestamp = frame.split('_')[0]

        if frame_type == "depth":
            depth_frame = cv2.imread(os.path.join(video_dir, frame), cv2.IMREAD_UNCHANGED)
            depth_timestamp = int(timestamp)

        elif frame_type == "rgb":
            color_frame = cv2.imread(os.path.join(video_dir, frame))
            color_timestamp = int(timestamp)

        if depth_timestamp == color_timestamp:
            pedestrians = detector.detect_pedestrians(color_frame, depth_frame)
            if pedestrians is not None:
                color_frame = tracker.track_pedestrians(color_frame, depth_frame, pedestrians)

            cv2.imshow("Action Recognition", color_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break




if __name__ == "__main__":
    main()