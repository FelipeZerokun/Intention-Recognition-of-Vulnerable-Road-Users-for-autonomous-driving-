import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionRecognitionModel:
    def __init__(self, action_labels: dict, model_dir: str, device: str = "cpu", sequence_length: int = 16):

        self.device = device
        self.labels = action_labels
        self.num_classes = len(action_labels)
        self.sequence_length = sequence_length

        if model_dir is None:
            self.model = self.get_i3d_model(self.num_classes)
        else:
            self.model = self.load_ar_model(model_dir, device, self.num_classes)
    def get_i3d_model(num_classes:int = 2):
        model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
        model.blocks[-1].proj = nn.Linear(in_features=2048, out_features=num_classes)
        return model

    def load_ar_model(self, model_dir, device, num_classes:int = 2):
        # model = i3d_r50(pretrained=False)  # Same architecture as used during training
        model = torch.hub.load("facebookresearch/pytorchvideo", "i3d_r50", pretrained=True)
        model.blocks[-1].proj = nn.Linear(in_features=2048, out_features=num_classes)
        model.load_state_dict(torch.load(model_dir, map_location=device))
        return model

    def action_prediction(self, sequence_tensor):
        """
        Predict the action for a sequence of pedestrian frames.
        Args:
           model: Action recognition model.
           sequence_tensor: Tensor containing the sequence of frames.
        """

        with torch.no_grad():
            output = self.model(sequence_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_action_idx = torch.argmax(probabilities, dim=1)
            predicted_action = self.labels[predicted_action_idx.item()]
            predicted_likelihood = probabilities[0, predicted_action_idx].item()

        return predicted_action, predicted_likelihood