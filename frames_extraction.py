from project_utils import *
import cv2
import os
from pathlib import Path


class VideoLabeling:
    def __init__(self, videos_dir: Path, frames_dir: Path):
        """Class for annotating videos

        Args:
            video_path (str): Path to the video file
            output_dir (str): Path to the output directory

        """

        self.videos_dir = videos_dir
        check_path(folder_path=self.videos_dir, create=False)
        self.frames_dir = frames_dir
        check_path(folder_path=self.frames_dir, create=True)


    def extract_frames_single(self, frame_rate=5):
        """Extract frames from the video file

        Args:
            frame_rate (int): Frame rate for extracting frames

        """

        for test_folder in os.listdir(self.videos_dir):
            test_folder_path = Path(self.videos_dir, test_folder)
            for video in os.listdir(test_folder_path):
                if video.endswith('.avi'):
                    video_name = video.split('.')[0]
                    video_path = Path(test_folder_path, video)
                    output_frame_path = Path(self.frames_dir, test_folder, "frames")
                    check_path(output_frame_path, create=True)
                    print(f"Extracting frames from video: {video_name} in folder {test_folder}")
                    cap = cv2.VideoCapture(str(video_path))
                    success, image = cap.read()
                    count = 0
                    while success:
                        if count % frame_rate == 0:
                            frame_name = f"{video_name}_frame_{count}.jpg"
                            frame_path = Path(output_frame_path, frame_name)
                            print(frame_path)
                            cv2.imwrite(str(frame_path), image)
                        success, image = cap.read()
                        count += 1
                    cap.release()

def main():
    # data_dir = os.environ.get('DATA_DIR')
    data_dir = Path('/media/felipezero/T7 Shield/DATA/thesis/videos')
    videos_dir = Path(data_dir, 'unlabeled_videos')
    output_dir = Path(data_dir, 'labeled_videos')
    video_labeling = VideoLabeling(videos_dir, output_dir)
    video_labeling.extract_frames_single()


if __name__ == "__main__":
    main()
