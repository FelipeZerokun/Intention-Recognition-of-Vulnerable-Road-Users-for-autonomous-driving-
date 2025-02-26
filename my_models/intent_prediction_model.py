import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


from project_utils.project_utils import check_path
class IntentPredictionModel:
    def __init__(self, predicted_labels: dict, model_dir: str, device: str = "cpu", sequence_length: int = 15):

        self.device = device
        self.labels = predicted_labels
        self.num_classes = len(self.labels)
        self.sequence_length = sequence_length
        check_path(model_dir)


        self.model = load_model(model_dir, compile=False)


def main():
    model_dir = '../PIEPredict-master/data/pie/intention/context_loc_pretrained/model.h5'
    predict_labels = {0: 'standing_still', 1: ''}
    model = IntentPredictionModel(predicted_labels = {0: 'Crossing', 1: 'Not Crossing'}, model_dir = model_dir)

if __name__ == '__main__':
    main()