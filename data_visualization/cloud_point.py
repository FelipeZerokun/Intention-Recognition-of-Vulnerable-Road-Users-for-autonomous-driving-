import open3d as o3d
import os
import time
from pathlib import Path

class CloudpointVisualizer:
    """Visualize point cloud data in 3D space"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def get_points_from_dir(self):
        """
        Loop through all files in the directory and load the PLY files.
        """
        point_clouds = {}
        for file in os.listdir(self.data_dir):
            if file.endswith('.ply'):
                file_path = os.path.join(self.data_dir, file)
                timestamp = file_path.split('_')[0]
                point_cloud = self.load_ply_file(file_path)
                point_clouds[timestamp] = point_cloud

        return self.order_point_clouds(point_clouds)

    def order_point_clouds(self, point_clouds):
        """
        Order the files by timestamp.
        """
        ordered_files = dict(sorted(point_clouds.items()))

        return ordered_files


    def load_ply_file(self, file_path: str):

        pc = o3d.io.read_point_cloud(Path(file_path))

        return pc


    def visualize_point_cloud(self):
        """
        Visualize the loaded point clouds one by one.

        Args:
            point_clouds (list): List of tuples containing file names and point cloud objects to visualize.
        """

        points = self.get_points_from_dir()
        for timestamp, point_cloud in points.items():
            print(f"Visualizing point cloud at timestamp {timestamp}")
            o3d.visualization.draw_geometries([point_cloud])
            time.sleep(100)

if __name__ == '__main__':
    data_dir = '../../DATA/video_01/'
    visualizer = CloudpointVisualizer(data_dir)
    visualizer.visualize_point_cloud()
