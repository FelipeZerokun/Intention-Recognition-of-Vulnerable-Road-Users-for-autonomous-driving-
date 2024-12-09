import torch
import torch.nn as nn
from torchvision import transforms

from torch.utils.data import DataLoader

from datasets.action_recognition_dataset import ActionRecognitionDataset

def get_i3d_model(num_classes:int = 2):
    model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
    model.blocks[-1].proj = nn.Linear(in_features=2048, out_features=num_classes)
    return model
