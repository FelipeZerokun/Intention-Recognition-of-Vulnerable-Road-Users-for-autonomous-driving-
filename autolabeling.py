import torch
import cv2
import os

from pathlib import Path
from utils import (check_path, check_file)

class IntentLabeler:

    def __init__(self, frames_dir: Path, labels_dir: Path):

        if check_path(frames_dir):
            self.frames_dir = frames_dir
        if check_path(labels_dir):
            self.labels_dir = labels_dir

        # Load YOLOv5 model
        self.model = torch.hub.load('yolov5', 'yolov5s', pretrained=True)


    def detect_and_label(self):
        """Detect and label objects in the video

        Args:
            frame_rate (int): Frame rate for extracting frames

        """

        for test_folder in os.listdir(self.frames_dir):
            test_folder_path = Path(self.frames_dir, test_folder)
            for frame in os.listdir(test_folder_path):
                if frame.endswith('.jpg'):
                    frame_name = frame.split('.')[0]
                    frame_path = Path(test_folder_path, frame)
                    output_label_path = Path(self.labels_dir, test_folder, "labels")
                    check_path(output_label_path, create=True)
                    print(f"Detecting objects in frame: {frame_name} in folder {test_folder}")
                    results = self.model(frame_path)
                    results.save(Path(output_label_path, frame_name), data=results.pandas().xyxy[0])



def main():
    print("Test for auto labeling")

if __name__ == '__main__':
    main()