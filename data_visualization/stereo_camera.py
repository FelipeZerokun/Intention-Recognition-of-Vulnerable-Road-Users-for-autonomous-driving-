import cv2
import pandas as pd
import numpy as np
from project_utils.project_utils import check_os_windows

class StereocameraVisualizer:
    """
    A class for visualizing both the RBG images and Depth images extracted from a stereo camera.

    Args:
        video_csv_data (str): Path to the directory of the video data file.
        camera_info_dir (str): Path to the camera info file.
    """
    def __init__(self, video_csv_data: str, camera_info_dir: str):
        self.frame_data_dir = video_csv_data
        self.camera_info_dir = camera_info_dir

        self.image_size, self.intrinsic_matrix = self.get_camera_info()

        self.color_frames, self.depth_frames, self.timestamps = self.load_frame_data()


    def get_camera_info(self):
        """
        Get the camera info from the CSV file.
        """
        camera_info = pd.read_csv(self.camera_info)
        image_height = camera_info.loc[0, 'image_height']
        image_width = camera_info.loc[0, 'image_width']

        intrinsic_matrix_str = camera_info.loc[0, 'Intrinsic_camera_matrix']

        # Convert the string representation of the matrix to a numpy array
        intrinsic_matrix = np.fromstring(intrinsic_matrix_str.strip('[]'), sep=' ').reshape((3, 3))

        return ([image_height, image_width], intrinsic_matrix)

    def load_frame_data(self):

        frame_data = pd.read_csv(self.frame_data_dir)
        color_frames = frame_data['rgb_frame'].tolist()
        depth_frames = frame_data['depth_frame'].tolist()
        timestamps = frame_data['timestamp'].tolist()

        return color_frames, depth_frames, timestamps

    def combine_rgb_depth_frames(self, frame, depth_map, depth_threshold=10000):
        """
        Combine a RGB and a Depth Map into a single image.
        """
        # if values from the depth map are greater than the threshold, make them 0
        depth_map[depth_map > depth_threshold] = 0

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_HOT)
        combined = cv2.addWeighted(frame, 0.7, depth_colormap, 0.3, 0)

        return combined

    def merge_images(self, image1, image2, image3):
        """
        Merge the RGB and Depth images into a single image.
        Args:
            image1 (numpy.ndarray): RGB image.
            image2 (numpy.ndarray): Depth image.
            image3 (numpy.ndarray): Additional image to merge with.
        """

        # Merge the images horizontally
        merged_image = cv2.hconcat([image1, image2, image3])

        return merged_image

    def visualize_stereo_camera(self):
        """
        Visualize the stereo camera images.
        """

        index = 0
        while True:
            frame_path = self.color_frames[index]
            depth_path = self.depth_frames[index]
            frame_path = check_os_windows(frame_path)
            depth_path = check_os_windows(depth_path)

            frame = cv2.imread(frame_path)
            depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

            combined_images = self.combine_rgb_depth_frames(frame, depth_map, depth_threshold=10000)
            merged_images = self.merge_images(frame, depth_map, combined_images)

            cv2.imshow('Frame viewer', merged_images)

            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC key
                break

            if key == ord('a'):
                if index == 0:
                    print('Already at the first frame.')
                    continue
                else:
                    index -= 1
                    continue

            if key == ord('d'):
                if index == len(self.color_frames) - 1:
                    print('Already at the last frame.')
                    continue
                else:
                    index += 1
                    continue

            if key == ord('q'):
                break



