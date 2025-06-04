import cv2
import numpy as np

import open3d as o3d
import sensor_msgs.point_cloud2 as pc2
from scipy.spatial.transform import Rotation as R

def get_camera_info(camera_info_msg):
    """
    Extract camera intrinsic parameters from a camera info message.

    Args:
        camera_info_msg (sensor_msgs.msg.CameraInfo): The ROS camera info message containing the camera parameters.

    Returns:
        tuple: A tuple containing the focal length (fx, fy), and the principal point (cx, cy).
    """
    fx = camera_info_msg.K[0]
    fy = camera_info_msg.K[4]
    cx = camera_info_msg.K[2]
    cy = camera_info_msg.K[5]

    return fx, fy, cx, cy

def get_odometry(odom_msg):
    """
    Extracts the robot's position and commands from an odometry message.

    Args:
        odom_msg (nav_msgs.odom_msg): A ROS odometry message containing pose and twist information.

    Returns:
        tuple: A tuple containing:
            - robot_position (np.ndarray): An array with the robot's x and y positions and yaw angle ([x_pos, y_pos, yaw]).
            - robot_commands (np.ndarray): An array with the robot's linear velocity and angular velocity ([x_vel, z_rot]).
    """
    x_pos = odom_msg.pose.pose.position.x
    y_pos = odom_msg.pose.pose.position.y

    quaternion = (
        odom_msg.pose.pose.orientation.x,
        odom_msg.pose.pose.orientation.y,
        odom_msg.pose.pose.orientation.z,
        odom_msg.pose.pose.orientation.w
    )

    r = R.from_quat(quaternion)
    roll, pitch, yaw = r.as_euler('xyz', degrees=False)

    x_vel = odom_msg.twist.twist.linear.x
    z_rot = odom_msg.twist.twist.angular.z

    robot_position = np.array([x_pos, y_pos, yaw])

    robot_commands = np.array([x_vel, z_rot])

    return robot_position, robot_commands

def get_llh_position(gps_msg):
    """
    Extracts latitude, longitude, and height from a GPS message.

    Args:
        gps_msg (geometry_msgs/PointStamped): A ROS GPS message containing position information.

    Returns:
        tuple: A tuple containing:
            - latitude (float): The latitude value from the GPS message.
            - longitude (float): The longitude value from the GPS message.
            - height (float): The height value from the GPS message.
    """
    latitude = gps_msg.point.x
    longitude = gps_msg.point.y
    height = gps_msg.point.z

    return latitude, longitude, height

def get_NED_position(gps_msg):
    """
    Extracts the North, East, and Down (NED) position from a GPS message.

    Args:
        gps_msg (geometry_msgs/PointStamped): A ROS GPS message containing position information.

    Returns:
        tuple: A tuple containing:
            - x_pos (float): The North position (x-coordinate) from the GPS message.
            - y_pos (float): The East position (y-coordinate) from the GPS message.
            - z_pos (float): The Down position (z-coordinate) from the GPS message.
    """
    x_pos = gps_msg.point.x
    y_pos = gps_msg.point.y
    z_pos = gps_msg.point.z

    return x_pos, y_pos, z_pos

def get_ECEF_position(gps_msg):
    """
    Extracts the Earth-Centered, Earth-Fixed (ECEF) position from a GPS message.

    Args:
        gps_msg (geometry_msgs/PointStamped): A ROS GPS message containing position information.

    Returns:
        tuple: A tuple containing:
            - x_pos (float): The x-coordinate in the ECEF coordinate system.
            - y_pos (float): The y-coordinate in the ECEF coordinate system.
            - z_pos (float): The z-coordinate in the ECEF coordinate system.
    """
    x_pos = gps_msg.point.x
    y_pos = gps_msg.point.y
    z_pos = gps_msg.point.z

    return x_pos, y_pos, z_pos

def get_velocity(vel_msg):
    """
    Extracts the linear and angular velocity from a velocity message.

    Args:
        vel_msg (geometry_msgs/Vector3Stamped): A ROS velocity message containing vector information.

    Returns:
        tuple: A tuple containing:
            - x_vel (float): The linear velocity in the x-direction.
            - y_vel (float): The linear velocity in the y-direction.
            - z_vel (float): The angular velocity in the z-direction.
    """
    x_vel = vel_msg.vector.x
    y_vel = vel_msg.vector.y
    z_vel = vel_msg.vector.z

    return x_vel, y_vel, z_vel

def get_color_image(img_msg):
    """
    Converts a ROS image message to an OpenCV image.

    Args:
        img_msg (sensor_msgs/Image): A ROS image message containing image data.

    Returns:
        np.ndarray: An OpenCV image in RGB format.
    """
    frame = np.frombuffer(img_msg.data, dtype=np.uint8)

    frame = frame.reshape(img_msg.height, img_msg.width, 3)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def get_depth_image(depth_msg, fix_frame=True):
    """
    Converts a ROS depth image message to an OpenCV image with a colormap.

    Args:
        depth_msg (sensor_msgs/Image): A ROS depth image message containing depth data.
        fix_frame (bool, optional): Whether to apply a transformation to the depth frame. Defaults to True.

    Returns:
        np.ndarray: An OpenCV image representing the depth data.
    """
    frame = np.frombuffer(depth_msg.data, dtype=np.uint16)
    frame = frame.reshape(depth_msg.height, depth_msg.width, 1)

    if fix_frame:
        frame = convert_depth2image(frame)

    return frame

def convert_depth2image(depth_array):
    """
    Crops and resizes a depth array to match the original dimensions.

    Args:
        depth_array (np.ndarray): A 2D array representing the depth data.

    Returns:
        np.ndarray: The cropped and resized depth array.
    """

    cropped_image = depth_array[120:-120, 120:-120]
    cropped_image = cv2.resize(cropped_image, (depth_array.shape[1], depth_array.shape[0]))

    return cropped_image

def visualize_data(color_image, depth_image):
    """
    Visualizes the color and depth images by combining them with a colormap.

    Args:
        color_image (np.ndarray): The color image in RGB format.
        depth_image (np.ndarray): The depth image in grayscale format.

    Returns:
        None
    """
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_HOT)
    added_images = cv2.addWeighted(color_image, 0.7, depth_colormap, 0.3, 0)
    merged_images = cv2.hconcat([color_image, added_images])

    cv2.imshow("Original images", merged_images)
    cv2.waitKey(5000)

def get_pointcloud_map(pointcloud_msg, image_height, image_width):
    """
    Processes a ROS point cloud message to extract 3D points and RGB data, visualize the point cloud,
    and convert it to a 2D OpenCV image.

    Args:
        pointcloud_msg (sensor_msgs/PointCloud2): A ROS point cloud message containing 3D points and RGB data.
        image_height (int): Height of the output image.
        image_width (int): Width of the output image.

    Returns:
        np.ndarray: A numpy array containing the 3D points (XYZ) and RGB data.
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


