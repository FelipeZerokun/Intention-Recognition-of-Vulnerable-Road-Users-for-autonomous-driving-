import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API

from pathlib import Path


class StereoCameraManager:
    def __init__(self, rosbag_path: str):
        print(rosbag_path)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device_from_file()
        self.profile = self.pipeline.start(self.config)
        # Skip 5 first frames to give the Auto-Exposure time to adjust

        for x in range(5):
            self.pipeline.wait_for_frames()

        print("Camera setup completed")


    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        self.pipeline.stop()
        print("Frames retrieved")
        return color_frame, depth_frame

    def get_color_image(self, color_frame):
        color_image = np.asanyarray(color_frame.get_data())
        plt.rcParams["axes.grid"] = False
        plt.rcParams['figure.figsize'] = [12, 6]
        plt.imshow(color_image)

    def get_depth_image(self, depth_frame):
        colorizer = rs.colorizer()
        colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())
        plt.imshow(colorized_depth)

    def align_frames(self, color_frame, depth_frame):
        align = rs.align(rs.stream.color)
        frames = align.process(rs.frame)
        aligned_depth_frame = frames.get_depth_frame()
        colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
        plt.imshow(colorized_depth)

def main():
    rosbag_path = '/media/felipezero/T7 Shield/DATA/thesis/Rosbags/2023_05_05/'
    rosbag_name = 'Test3_12_37_C-R/2023_05_05_12_37_Gera_C-R_Alt.orig.bag'

    cam_manager = StereoCameraManager(rosbag_path + rosbag_name)

if __name__ == '__main__':
    main()