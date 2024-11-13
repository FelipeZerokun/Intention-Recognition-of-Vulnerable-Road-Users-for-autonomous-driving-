import cv2
import numpy as np

import pyrealsense2 as rs
import open3d as o3d

from sensor_msgs.msg import PointCloud2
import tf.transformations
import sensor_msgs.point_cloud2 as pc2

def get_camera_info(camera_info_msg):
    """Get camera intrinsic parameters from camera info message
    """
    fx = camera_info_msg.K[0]
    fy = camera_info_msg.K[4]
    cx = camera_info_msg.K[2]
    cy = camera_info_msg.K[5]

    return fx, fy, cx, cy

def get_odometry(odom_msg):
    """Get odometry from odometry message
    """
    x_pos = odom_msg.pose.pose.position.x
    y_pos = odom_msg.pose.pose.position.y
    z_pos = odom_msg.pose.pose.position.z

    quaternion = (
        odom_msg.pose.pose.orientation.x,
        odom_msg.pose.pose.orientation.y,
        odom_msg.pose.pose.orientation.z,
        odom_msg.pose.pose.orientation.w
    )

    x_vel = odom_msg.twist.twist.linear.x
    y_vel = odom_msg.twist.twist.linear.y
    z_vel = odom_msg.twist.twist.linear.z

    return x_pos, y_pos, z_pos, x_vel, y_vel, z_vel

def get_llh_position(gps_msg):
    """Get latitude, longitude and height from GPS message
    """
    latitude = gps_msg.point.x
    longitude = gps_msg.point.y
    height = gps_msg.point.z

    return latitude, longitude, height

def get_NED_position(gps_msg):
    """Get North, East and Down position from GPS message
    """
    x_pos = gps_msg.point.x
    y_pos = gps_msg.point.y
    z_pos = gps_msg.point.z

    return x_pos, y_pos, z_pos

def get_ECEF_position(gps_msg):
    """Get Earth-Centered, Earth-Fixed position from GPS message
    """
    x_pos = gps_msg.point.x
    y_pos = gps_msg.point.y
    z_pos = gps_msg.point.z

    return x_pos, y_pos, z_pos
def get_color_image(img_msg):
    """Convert ROS image message to OpenCV image
    """
    frame = np.frombuffer(img_msg.data, dtype=np.uint8)

    frame = frame.reshape(img_msg.height, img_msg.width, 3)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def get_depth_image(depth_msg):
    """Convert ROS depth image message to OpenCV image with a colormap

    """

    frame = np.frombuffer(depth_msg.data, dtype=np.uint16)
    frame = frame.reshape(depth_msg.height, depth_msg.width, 1)

    # Cropping the image to make it align with the RGB image
    croppeg_image = frame[120:-120, 120:-120]

    rescaled_image = cv2.resize(croppeg_image, (depth_msg.width, depth_msg.height))

    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(rescaled_image, alpha=0.03), cv2.COLORMAP_HOT)
    merged_images = cv2.hconcat([frame, rescaled_image])
    return depth_colormap
def get_depth_map(depth_msg, image_height, image_width):
    """Convert ROS depth image message to OpenCV image
    """
    depth_map = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(image_height, image_width, 1)
    depth_map_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return depth_map_norm

def overlap_depth_map_with_color_image(color_image, depth_image):
    """Overlap depth map with color image
    """
    added_images = cv2.addWeighted(color_image, 0.7, depth_image, 0.3, 0)
    merged_images = cv2.hconcat([color_image, depth_image])

    cv2.imshow("Original images", merged_images)
    cv2.imshow("Added images", added_images )
    cv2.waitKey(5000)

    return added_images

def align_depth_image_with_color_image(images):
    """Align depth image with color image
    """
    pipe = rs.pipeline()
    cfg = rs.config()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(images)
    return aligned_frames



def get_pointcloud_map(pointcloud_msg, image_height, image_width):
    """Convert ROS pointcloud message to OpenCV image
    """

    height = pointcloud_msg.height
    width = pointcloud_msg.width
    dtype_list = []

    pointcloud_data = pc2.read_points(pointcloud_msg, field_names=("x", "y", "z","rgb"), skip_nans=True)
    points = np.array(list(pointcloud_data), dtype=np.float32)

    # Extract XYZ and RGB components
    xyz = points[:, :3]
    rgb = points[:, 3].view(np.uint32)
    rgb = np.stack(((rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF), axis=-1) / 255.0

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)

    # Visualize point cloud
    o3d.visualization.draw_geometries([pcd])

    # Convert Open3D point cloud to OpenCV image
    image = pointcloud_to_image(xyz, rgb, image_height, image_width)

    #cv2.imshow('Point cloud', image)
    #cv2.waitKey(5000)

    return points

def pointcloud_to_image(xyz, rgb, image_height, image_width):
    """
    Convert point cloud data to a 2D image.

    Args:
        xyz (np.ndarray): Array of 3D coordinates (N, 3).
        rgb (np.ndarray): Array of RGB values (N, 3).
        image_height (int): Height of the output image.
        image_width (int): Width of the output image.

    Returns:
        np.ndarray: 2D image created from the point cloud data.
    """
    # Initialize an empty image
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    # Normalize the xyz coordinates to fit within the image dimensions
    xyz_normalized = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))
    xyz_normalized[:, 0] *= image_width
    xyz_normalized[:, 1] *= image_height

    # Populate the image with the RGB values
    for i in range(xyz.shape[0]):
        x = int(xyz_normalized[i, 0])
        y = int(xyz_normalized[i, 1])
        if 0 <= x < image_width and 0 <= y < image_height:
            image[y, x, :] = (rgb[i] * 255).astype(np.uint8)

    return image

