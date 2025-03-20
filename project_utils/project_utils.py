import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import ast

def check_os_windows(data_dir: str|Path):
    """Check if the OS is Windows and change the path accordingly

    Args:
        data_dir (str): Path to change

    Returns:
        Path: Path with the correct format
    """
    if os.name == 'nt':
        data_dir = str(data_dir).replace('\\', '/')
        data_dir = data_dir.replace('/media/felipezero/T7 Shield/', 'E:/')
        return Path(data_dir)
    else:
        return data_dir
def check_path(folder_path, create=False):
    """Check if the path exists, if not create it

    Args:
        path (str): Path to check

    Returns:
        True if the path exists
    """
    if not os.path.exists(folder_path):
        print(f'Path {folder_path} does not exist.')
        if create:
            os.makedirs(folder_path)
            print(f'Path {folder_path} created successfully.')
            return True
        else:
            raise FileNotFoundError

    else:
        print(f'Path {folder_path} exists.')
        return True


def check_file(path_to_file):
    """Check if the file exists

    Args:
        path_to_file (str): Path to the file

    Returns:
        True if the file exists
    """
    if not os.path.exists(path_to_file):
        print(f'File {path_to_file} does not exist.')
        return False
    else:

        print(f'File {path_to_file} exists.')
        return True


def save_image_file(image, image_path, depth=False):
    """Save an image to a file

    Args:
        image (np.array): Image to save
        image_path (str): Path to save the image
    """
    if depth:
        cv2.imwrite(image_path, image.astype(np.uint16))

    else:
        cv2.imwrite(image_path, image)

def save_csv_file(data, file_name, output_dir, create=True):
    """Save data to a CSV file

    Args:
        data (dict or pd.DataFrame): Data to save or to append

    """

    action = data['action']
    intent = data['intent']
    track_id = data['track_id']
    pedestrian_dir = os.path.join(output_dir, f'{action}_{intent}/pedestrian_{track_id}/')

    if create:
        check_path(folder_path=pedestrian_dir, create=True)

        pedestrian_data_df = pd.DataFrame([data])
        pedestrian_data_df.to_csv(os.path.join(pedestrian_dir, f'pedestrian_{track_id}.csv'), index=False)
        pedestrian_data_df.to_csv(file_name, index=False)

        print(f"Data from pedestrian {track_id} saved successfully in the directory {pedestrian_dir}.")

    else:
        pedestrian_file_path = os.path.join(pedestrian_dir, f'pedestrian_{track_id}.csv')
        check_path(folder_path=pedestrian_file_path, create=False)

        pedestrian_data_df = pd.read_csv(pedestrian_file_path)
        pedestrian_data_df = pd.concat([pedestrian_data_df, pd.DataFrame([data])], ignore_index=True)
        pedestrian_data_df.to_csv(pedestrian_file_path, index=False)
        pedestrian_data_df.to_csv(file_name, index=False)

    return pedestrian_dir


def parse_position_string(position_str):
    # for data_string in position_str.split():
    #     if data_string.startswith('['):
    #         position_str = data_string
    #         break
    position_list = ast.literal_eval(position_str)  # Convert string to list
    return tuple(map(float, position_list))  # Convert all elements to float and return as tuple


def get_data(data, mode: str):
    if mode == 'NED':
        ned_data = parse_position_string(data)
        North = round(ned_data[0], 3)
        East = round(ned_data[1], 3)
        Down = round(ned_data[2], 3)
        return [North, East, Down]

    elif mode == 'ECEF':
        ecef_data = parse_position_string(data)
        x_positions, y_positions, z_positions = zip(*ecef_data)
        return x_positions, y_positions, z_positions

    elif mode == 'LLH':
        llh_data = parse_position_string(data)
        latitudes, longitudes, heights = zip(*llh_data)

        return latitudes, longitudes, heights

    elif mode == 'velocity':
        vel_data = parse_position_string(data)
        x_velocity = round(vel_data[0], 6)
        y_velocity = round(vel_data[1], 6)
        z_velocity = round(vel_data[2], 6)
        return x_velocity, y_velocity, z_velocity

def estimate_pedestrian_distance(depth_image, depth_scale=0.001):
    """Restore the depth values to their original range

    Args:
        depth_image (np.array): Depth image
        min_max_depth (tuple): Min and max depth values
        depth_scale (float): Depth scale

    Returns:
        np.array: Depth image with restored values
    """

    mean_depth = np.mean(depth_image)
    # Get a small 5x5 matrix in the center of the image and calculate the mean of that matrix
    # This is done to avoid considering the background
    center_x = depth_image.shape[1] // 2
    center_y = depth_image.shape[0] // 2
    mean_depth_around_center = np.mean(depth_image[center_y-10:center_y+11, center_x-10:center_x+11])

    # print(f"Comparing methods to obtain the distance")
    # print(f"First with regular mean: {mean_depth*depth_scale}")
    # print(f"Second with mean around center: {mean_depth_around_center*depth_scale}")
    return mean_depth_around_center * depth_scale

def estimate_pedestrian_position(bbox, pedestrian_distance, intrinsic_matrix):
    """Estimate the pedestrian position in 3D space using the intrinsic matrix of the camera.
    Args:
        bbox (tuple): Bounding box coordinates (x1, y1, x2, y2)
        pedestrian_distance (float): Estimated distance to the pedestrian
        intrinsic_matrix (np.array): Intrinsic matrix of the camera
    Returns:
        tuple: Estimated 3D position of the pedestrian (X, Y, Z)
    """
    fx = intrinsic_matrix[0, 0] # focal length
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2] # optical center
    cy = intrinsic_matrix[1, 2]

    # Get the center of the bounding box
    u = (bbox[0] + bbox[2]) // 2
    v = (bbox[1] + bbox[3]) // 2

    # Convert to 3D coordinates
    X = (u - cx) * pedestrian_distance / fx
    Y = (v - cy) * pedestrian_distance / fy

    return [X, Y, pedestrian_distance]


def estimate_pedestrian_speed(distance, time_interval, robot_speed):
    """Estimate the pedestrian speed

    Args:
        distance (float): Distance to the pedestrian
        time_interval (float): Time interval between frames
        robot_speed (float): Robot speed

    Returns:
        float: Estimated pedestrian speed
    """
    pedestrian_speed = (distance / time_interval)

    final_speed = pedestrian_speed + robot_speed


    return pedestrian_speed

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)