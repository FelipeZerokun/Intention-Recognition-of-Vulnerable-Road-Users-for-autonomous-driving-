import rosbags
import cv_bridge

from pathlib import Path
class Rosbags_Manager():
    def __init__(self, rosbag_path: Path ):
        self.rosbag_path = rosbag_path

    def read_rosbag(self):
        rosbag = rosbags.Bag(self.rosbag_path)
        for topic in rosbag.topics:
            print(topic)

