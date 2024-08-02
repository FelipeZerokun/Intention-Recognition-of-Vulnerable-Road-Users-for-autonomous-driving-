import os

def check_path(folder_path, create=False):
    """Check if the path exists, if not create it

    Args:
        path (str): Path to check
    """
    if not os.path.exists(folder_path):
        print(f'Path {folder_path} does not exist.')
        if create:
            os.makedirs(folder_path)
            print(f'Path {folder_path} created successfully.')
        else:
            raise FileNotFoundError

    else:
        print(f'Path {folder_path} exists.')


def check_file(path_to_file):
    """Check if the file exists

    Args:
        path_to_file (str): Path to the file
    """
    if not os.path.exists(path_to_file):
        print(f'File {path_to_file} does not exist.')
        raise FileNotFoundError
    else:
        print(f'File {path_to_file} exists.')