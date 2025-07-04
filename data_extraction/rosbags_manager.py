import rosbag
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import logging
import subprocess
from typing import Dict, List, Tuple

from project_utils.project_utils import (
    check_path,
    save_image_file,
)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class RosbagManager():
    """
        A class to manage and extract data from ROS bag files.

    """

    _stereo_info = '/camera/color/camera_info'
    _stereo_image = '/camera/color/image_raw'
    _depth_image = '/camera/depth/image_rect_raw'
    _depth_pointcloud = '/camera/depth/color/points'

    _depth_scale = 0.001

    _ned_pos_topic = '/anavs/solution/pos'
    _global_pos_topic = '/anavs/solution/pos_llh'
    _ecef_pos_topic = '/anavs/solution/pos_xyz'
    _local_vel_topic = '/anavs/solution/vel'

    _odom_topic = '/RosAria/pose'
    _estimated_speed = '/anavs/solution/vel'

    _gnss_topics = [_ned_pos_topic,
                   _global_pos_topic,
                   _ecef_pos_topic,
                   _local_vel_topic,
                   _odom_topic,]

    _stereo_topics = [_stereo_info, _stereo_image, _depth_image, _depth_pointcloud]

    _topics = [_odom_topic, *_stereo_topics, *_gnss_topics]

    def __init__(self, path: Path, name: str) -> None:
        """
        Initializes the RosbagManager object.

        Args:
            path (Path): Path to the directory containing the ROS bag file.
            name (str): Name of the ROS bag file.
        """
        self.path = path
        self.name = name
        self.file = Path(self.path, self.name)
        self.bag = None

    def _open_bag(self) -> None:
        """
            Opens the ROS bag file for reading.
        """
        self.bag = rosbag.Bag(self.file, 'r', skip_index=True)
        print("Rosbag opened successfully")
        logging.info('Rosbag opened successfully.')

    def _close_bag(self) -> None:
        """
            Closes the ROS bag file.
        """
        self.bag.close()
        logging.info('Rosbag closed successfully.')

    def reindex_bag(self) -> None:
        """"
            Reindexes the ROS bag file to fix any indexing issues.
        """

        try:
            logging.info(f'Reindexing rosbag {self.name}')
            print("Reindexing rosbag")
            result = subprocess.run(['rosbag', 'reindex', self.file], capture_output=True, text=True)
            logging.info(result.stdout)


            if result.stderr:
                logging.warning(result.stderr)

            logging.info(f'Rosbag {self.name} reindexed successfully.')
        except Exception as e:
            logging.warning(f'Error while reindexing rosbag {self.name}: {e}')

    def check_bag(self) -> None:
        """
            Checks the contents of the ROS bag file and prints metadata.
        """
        try:
            self._open_bag()
            print(f'Bag duration: {self.bag.get_end_time() - self.bag.get_start_time()}')
            print(f'Bag messages: {self.bag.get_message_count()}')

            info = self.bag.get_type_and_topic_info()
            topics = info.topics

            print("Info: {}".format(info))
            print(f"Number of topics: {len(topics)}")
            print("---------------------------------------------")

            for topic, metadata in topics.items():
                print(f"Topic: {topic}")
                print(f"Message type: {metadata.msg_type}")
                print(f"Message count: {metadata.message_count}")
                print(f"Frequency: {metadata.frequency}")
                print("---------------------------------------------")



        except Exception as e:

            logging.warning(f'Error while opening rosbag {self.name}: {e}')

        finally:
            self._close_bag()

    def extract_all_data(self, output_dir: str) -> None:
        """
        Extracts all relevant data from the ROS bag file and saves it to the specified directory.

        Args:
            output_dir (str): Directory to save the extracted data.
        """
        frames_to_skip = 1
        show_info = True
        depth_map_check = False
        stereo_image_check = False
        pointcloud_check = False
        odom_check = False
        t_image = 0
        t_depth = 0
        pointcloud = None

        try:
            self._open_bag()
            check_path(output_dir, create=True)
            navigation_data = {
                "timestamp": [],
                "Robot odometry": [],
                "Robot NED position": [],
                "Robot ECEF position": [],
                "Robot Global position": [],
                "Robot estimated velocity": [],
                "rgb_frame": [],
                "depth_frame": [],
                "action": []
            }

            for topic, msg, t in self.bag.read_messages(topics=self._topics):
                if show_info:
                    if topic == self._stereo_info:
                        frames_to_skip -= 1
                        if frames_to_skip == 0:
                            print(f'Stereo camera info: {msg}')
                            image_height = msg.height
                            image_width = msg.width
                            intrinsic_camera_matrix = np.array(msg.K).reshape(3,3) # 3x3 matrix
                            distortion_parameters = np.array(msg.D)
                            rotation_params = np.array(msg.R).reshape(3,3) # 3x3 matrix
                            projection_matrix = np.array(msg.P).reshape(3,4) # 3x4 matrix

                            # Initialize the variables for the robot position
                            x_pos = y_pos = z_pos = 0
                            lat = long = height = 0
                            ecef_x = ecef_y = ecef_z = 0
                            vel_x = vel_y = vel_z = 0

                            show_info = False
                else:

                    if topic == self._odom_topic:
                        robot_pos, robot_commands = get_odometry(msg)
                        odom_check = True

                    elif topic == self._ned_pos_topic:
                        # Extract position
                        x_pos, y_pos, z_pos = get_NED_position(msg)

                    elif topic == self._global_pos_topic:
                        # Extract position
                        lat, long, height = get_llh_position(msg)

                    elif topic == self._ecef_pos_topic:
                        # Extract position
                        ecef_x, ecef_y, ecef_z = get_ECEF_position(msg)

                    elif topic == self._local_vel_topic:
                        vel_x, vel_y, vel_z = get_velocity(msg)

                    elif topic == self._depth_image:
                        depth_image = get_depth_image(msg, fix_frame=False)
                        depth_map_check = True

                    elif topic == self._stereo_image:
                        # Extract frame
                        frame = get_color_image(msg)
                        timestamp = t.to_nsec()
                        stereo_image_check = True

                    elif topic == self._depth_pointcloud:
                        # points = get_pointcloud_map(msg, image_height, image_width)
                        pointcloud_check = True

                    if depth_map_check and stereo_image_check and odom_check:
                        # visualize_data(frame, depth_image)

                        color_file_name = output_dir + f'{timestamp}_rgb_frame.png'
                        depth_file_name = output_dir + f'{timestamp}_depth_frame.png'

                        save_image_file(frame, color_file_name)
                        save_image_file(depth_image, depth_file_name, depth=True)

                        navigation_data["timestamp"].append(timestamp)
                        navigation_data["Robot odometry"].append(robot_pos)
                        navigation_data["Robot NED position"].append([x_pos, y_pos, z_pos])
                        navigation_data["Robot ECEF position"].append([ecef_x, ecef_y, ecef_z])
                        navigation_data["Robot Global position"].append([lat, long, height])
                        navigation_data["Robot estimated velocity"].append([vel_x, vel_y, vel_z])
                        navigation_data["rgb_frame"].append(color_file_name)
                        navigation_data["depth_frame"].append(depth_file_name)
                        navigation_data["action"].append("")

                        depth_map_check = stereo_image_check = odom_check = False

            # Write the data into a CSV file
            nav_data_df = pd.DataFrame(navigation_data)
            csv_file_name = output_dir + 'navigation_data.csv'

            nav_data_df.to_csv(csv_file_name, index=False)

            print("Data extracted successfully.")


        except Exception as e:
            print(f'Error while extracting data from rosbag {self.name}: {e}')

        finally:
            print("closing rosbag")
            cv2.destroyAllWindows()
            self._close_bag()

    def extract_stereo_data(self, output_dir: str) -> None:
        """
        Extracts stereo data (RGB and depth frames) from the ROS bag file and saves it to the specified directory.

        This function processes stereo camera topics to extract RGB frames and depth maps.
        The extracted data is saved as image files, and metadata is stored in a CSV file.

        Args:
            output_dir (str): Directory to save the extracted RGB and depth frames, along with the metadata.

        Raises:
            Exception: If an error occurs during the extraction process.
        """
        frames_to_skip = 1
        show_info = True
        depth_map_check = False
        stereo_image_check = False
        t_image = 0
        t_depth = 0
        pointcloud = None

        try:
            self._open_bag()
            check_path(output_dir, create=True)
            navigation_data = {
                "timestamp": [],
                "rgb_frame": [],
                "depth_frame": [],
                "action": []
            }

            for topic, msg, t in self.bag.read_messages(topics=self._topics):
                if show_info:
                    if topic == self._stereo_info:
                        frames_to_skip -= 1
                        if frames_to_skip == 0:
                            print(f'Stereo camera info: {msg}')
                            image_height = msg.height
                            image_width = msg.width
                            intrinsic_camera_matrix = np.array(msg.K).reshape(3,3) # 3x3 matrix
                            distortion_parameters = np.array(msg.D)
                            rotation_params = np.array(msg.R).reshape(3,3) # 3x3 matrix
                            projection_matrix = np.array(msg.P).reshape(3,4) # 3x4 matrix

                            # Initialize the variables for the robot position
                            x_pos = y_pos = z_pos = 0
                            lat = long = height = 0
                            ecef_x = ecef_y = ecef_z = 0
                            vel_x = vel_y = vel_z = 0

                            show_info = False
                else:

                    if topic == self._depth_image:
                        depth_image = get_depth_image(msg)
                        depth_map_check = True

                    elif topic == self._stereo_image:
                        # Extract frame
                        frame = get_color_image(msg)
                        timestamp = t.to_nsec()
                        stereo_image_check = True

                    if depth_map_check and stereo_image_check:
                        # visualize_data(frame, depth_image)

                        color_file_name = output_dir + f'{timestamp}_rgb_frame.png'
                        depth_file_name = output_dir + f'{timestamp}_depth_frame.png'

                        save_image_file(frame, color_file_name)
                        save_image_file(depth_image, depth_file_name, depth=True)

                        navigation_data["timestamp"].append(timestamp)
                        navigation_data["rgb_frame"].append(color_file_name)
                        navigation_data["depth_frame"].append(depth_file_name)
                        navigation_data["action"].append("")

                        depth_map_check = stereo_image_check = False

            # Write the data into a CSV file
            nav_data_df = pd.DataFrame(navigation_data)
            csv_file_name = output_dir + 'navigation_data.csv'

            nav_data_df.to_csv(csv_file_name, index=False)

            print("Data extracted successfully.")


        except Exception as e:
            print(f'Error while extracting stereo images from rosbag {self.name}: {e}')

        finally:
            print("closing rosbag")
            cv2.destroyAllWindows()
            self._close_bag()

    def extract_video(self) -> None:
        """
        Extracts video frames from the ROS bag file and saves them as a video file.
        """
        try:
            self._open_bag()

            video_writer = None
            frame_count = 0

            for topic, msg, t in self.bag.read_messages(topics=['/camera/color/image_raw']):
                # Extract frame
                frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if video_writer is None:
                    # Initialize the video writer with the first frame's properties
                    fourcc = cv2.VideoWriter_fourcc(*'X264')  # Use X264 codec for MP4 format
                    fps = 30
                    output_path = Path(self.path, self.name.replace('.bag', '.mp4'))
                    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (msg.width, msg.height))

                # Write frame to video
                video_writer.write(frame)
                frame_count += 1

                # print(f'Frame {frame_count} extracted.')

            if video_writer is not None:
                video_writer.release()
                print(f'Video extracted successfully to {output_path}')


        except Exception as e:
            print(f'Error while extracting images from rosbag {self.name}: {e}')

        finally:
            self._close_bag()


def main():
    rosbag_path = '/internal/rosbags/'
    rosbag_name = '2023_05_05/video_01/2023_05_05_10_14_Gera_C-R_Alt.orig.bag'
    output_path = '/internal/rosbags/extracted_data/video_01/'
    print("Extracting data from", rosbag_name)
    bag = RosbagManager(rosbag_path, rosbag_name)
    # bag.extract_video()

    bag.check_bag()
    bag.extract_all_data(output_path)


if __name__ == '__main__':
    main()