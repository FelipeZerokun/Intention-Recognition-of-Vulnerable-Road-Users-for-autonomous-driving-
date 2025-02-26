import os
import imageio.v2 as imageio
from PIL import Image
import numpy as np

def save_pedestrian_action(pedestrian_class_dir: str):
    """
    Combines all the frames from the pedestrian's action into a GIF file to show the action.
    Args:
        pedestrian_class_dir (str): Path to the pedestrian's action directory

    """

    # Check all the frames inside the directory that end with .png extension
    frames = [f for f in os.listdir(pedestrian_class_dir) if f.endswith('.png')]
    pedestrian_id = pedestrian_class_dir.split('/')[-2]

    # Determine the size of the first image
    first_image_path = os.path.join(pedestrian_class_dir, frames[0])
    with Image.open(first_image_path) as img:
        size = img.size

    # Combines the frame into a gif file
    with imageio.get_writer(f'{pedestrian_class_dir}{pedestrian_id}.gif', mode='I') as writer:
        for frame in frames:
            image_path = os.path.join(pedestrian_class_dir, frame)
            with Image.open(image_path) as img:
                img_resized = img.resize(size, Image.LANCZOS)
                writer.append_data(np.array(img_resized))


if __name__ == "__main__":
    pedestrian_data_dir = 'E:/DATA/thesis/intent_prediction_dataset/classes_01/walking_1/pedestrian_1/'
    save_pedestrian_action(pedestrian_data_dir)
