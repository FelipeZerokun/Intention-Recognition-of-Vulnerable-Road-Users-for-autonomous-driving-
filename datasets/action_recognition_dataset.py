from torch.utils.data import Dataset
import os
import torch
from PIL import Image

class ActionRecognitionDataset(Dataset):

    """
    Args:
        data_dir (str): Directory containing the dataset.
        transform (callable, optional): Optional transform to be applied on a sample.
        sequence_length (int, optional): Length of the sequence to extract from each clip. Defaults to 16.
    """

    def __init__(self, data_dir, transform=None, sequence_length=16):
        self.data_dir = data_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.labels = {i: label for i, label in enumerate(os.listdir(data_dir))}

        self.data = self._load_samples()
        print("Dataset loaded successfully")

    def _load_samples(self):
        """
        Ladd th samples from the dataset directory.
        Each action class contains a directory with clips, each clip contains a directory with frames.
        """
        samples = []

        for label, action in enumerate(os.listdir(self.data_dir)):
            action_dir = os.path.join(self.data_dir, action)
            for clip in os.listdir(action_dir):
                clip_dir = os.path.join(action_dir, clip)

                frames = sorted([os.path.join(clip_dir, frame) for frame in os.listdir(clip_dir)])[1:]

                # Slip clips into 16-frames sub clips
                frames_len = len(frames)
                if frames_len >= self.sequence_length:
                    for i in range(0, frames_len - self.sequence_length + 1, self.sequence_length):
                        sub_clip = frames[i:i+self.sequence_length]
                        samples.append((sub_clip, label))

                    remaining_frames = frames_len % self.sequence_length
                    if remaining_frames > 0:
                        final_clip = frames[-self.sequence_length:]
                        samples.append((final_clip, label))

                else:
                    # Handle short clips by padding
                    padded_clip = frames + [frames[-1]] * (self.sequence_length - frames_len)
                    samples.append((padded_clip, label))

        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to load.

        Returns:
            tuple: (sequence, label)
        """
        clip_frames, label = self.data[idx]

        # Load the frames for the clip
        frames = [Image.open(frame_path).convert("RGB") for frame_path in clip_frames]

        # Apply transformations
        if self.transform:
            frames = [self.transform(frame) for frame in frames]

        # Stack frames into a tensor

        frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label
