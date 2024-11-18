import os
import cv2
import numpy as np
from pathlib import Path


def check_os_windows(data_dir: str):
    """Check if the OS is Windows and change the path accordingly

    Args:
        data_dir (str): Path to change

    Returns:
        Path: Path with the correct format
    """
    if os.name == 'nt':
        data_dir = str(data_dir).replace('\\', '/')
        data_dir = data_dir.replace('/media/felipezero/T7 Shield/', 'D:/')
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
        raise FileNotFoundError
    else:

        print(f'File {path_to_file} exists.')
        return True


def save_image_file(image, image_path):
    """Save an image to a file

    Args:
        image (np.array): Image to save
        image_path (str): Path to save the image
    """
    cv2.imwrite(image_path, image)

def restore_depth_values(depth_image, min_max_depth, depth_scale=0.001):
    """Restore the depth values to their original range

    Args:
        depth_image (np.array): Depth image
        min_max_depth (tuple): Min and max depth values
        depth_scale (float): Depth scale

    Returns:
        np.array: Depth image with restored values
    """

    restored_depth = cv2.normalize(depth_image, None, min_max_depth[0], min_max_depth[1], cv2.NORM_MINMAX)
    restored_depth = restored_depth.astype(np.uint16)

    return restored_depth