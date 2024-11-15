import os
import cv2

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