import torch
import pandas as pd
from pathlib import Path
import cv2
from project_utils import (check_path, check_file, )

#load YOLOv5 model
class PedestrianRecognition:
    def __init__(self, data_directory: Path):
        """Class for pedestrian recognition using YOLOv5 model
        """
        self.classes_dir = data_directory
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        self.detection_results = []

    def get_class(self):
        """ Checks if the directory has several floders representing the classes and
        if there is a csv file with the data of each frame"""

        detected_class = self.classes_dir.parts[-1]
        check_path(folder_path=self.classes_dir, create=False)

        print("Class found: ", detected_class)
        class_data = Path(self.classes_dir, f"{detected_class}.csv")
        check_file(class_data)
        class_data = pd.read_csv(class_data)

        return class_data



    def pedestrian_detection(self):
        """Apply YOLOv5 model to detect pedestrians in the frames
        """

        frame_data = self.get_class()


        for index, row in frame_data.iterrows():
            timestamp = row['timestamp']
            odometry = [row['x_pos'], row['y_pos'], row['z_pos'], row['x_vel'], row['y_vel'], row['z_vel']]
            frame = cv2.imread(row['frame'])

            results = self.model(frame)
            persons = results.xyxy[0][results.xyxy[0][:, 5] == 0]

            if len(persons) == 0:
                cv2.imshow('Pedestrian detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    continue

            for person in persons:
                x1, y1, x2, y2, confidence, class_id = person.tolist()
                detection = {
                    'timestamp': timestamp,
                    'odometry': odometry,
                    'bounding_box': [x1, y1, x2, y2],
                    'confidence': confidence
                }

                self.detection_results.append(detection)

            for person in persons:
                x1, y1, x2, y2, confidence, class_id = person
                if confidence < 0.6:
                    continue
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f'Person {confidence:.2f}', (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow('Pedestrian detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()



def main():

    frames_path = Path('/media/felipezero/T7 Shield/DATA/thesis/Videos/frames/classes/')
    class_01 = Path(frames_path / 'walking_2')

    pedestrian_recognition = PedestrianRecognition(data_directory=class_01)

    pedestrian_recognition.pedestrian_detection()

if __name__ == '__main__':
    main()
