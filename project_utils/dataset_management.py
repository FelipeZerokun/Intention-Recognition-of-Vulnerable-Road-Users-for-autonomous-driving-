import cv2
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
from datasets.action_recognition_dataset import ActionRecognitionDataset

def split_dataset(dataset_dir: str, clip_lenght=16, split_ratios=(0.7, 0.15, 0.15), seed=42):
    """
    Split the dataset into train, validation, and test sets.

    Args:
        dataset_dir (str): Path to the dataset directory.
        clip_length (int, optional): Length of the sequence to extract from each clip. Defaults for I3D is 16.
        split_ratios (tuple, optional): Ratios for train, validation, and test sets. Defaults to (0.7, 0.15, 0.15).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: Datasets for training, validation and test
    """
    assert sum(split_ratios) == 1.0, "Split ratios must sum to 1.0"

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

    # Initialize the dataset
    full_dataset = ActionRecognitionDataset(
        data_dir=dataset_dir,
        transform=transform,
        sequence_length=clip_lenght
    )

    # Calculate sizes for each split
    train_size = int(split_ratios[0] * len(full_dataset))
    val_size = int(split_ratios[1] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return train_dataset, val_dataset, test_dataset
