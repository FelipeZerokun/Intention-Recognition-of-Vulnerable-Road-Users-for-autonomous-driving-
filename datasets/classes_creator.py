from pathlib import Path
import shutil

import cv2
import pandas as pd

from project_utils.project_utils import check_path, check_file, check_os_windows, estimate_pedestrian_distance


class HumanActionClassCreation:
    def __init__(self, frames_data: Path, output_folder: Path, check_intention: False):
        """
        Class to create a dataset with human actions from several frames extracted from a video file.

        Args:
            video_data (Path): Path to the CSV file with the frame's data.

        """
        self._DATA_COLUMNS = ['timestamp', 'Robot odometry', 'rgb_frame', 'depth_frame']
        self.check_intention = check_intention
        self.frames_data = check_os_windows(frames_data)
        check_file(self.frames_data)

        self.output_folder = check_os_windows(output_folder)
        check_path(folder_path=self.output_folder, create=True)

        self.action_counter = 0

        self.create_classes()

    def check_correct_frames_data(self):
        """
        Check if the frames data CSV file has the correct columns.

        Raises:
            ValueError: If the columns are not correct.
        """
        frame_data = pd.read_csv(self.frames_data)

        columns = frame_data.columns

        for column in self._DATA_COLUMNS:
            if column not in columns:
                raise ValueError(f'Column {column} not found in the frames data CSV file.')

        return frame_data

    def create_classes(self):
        """
        Manual creation of classes for human actions in the frames.
        """
        frame_data = self.check_correct_frames_data()
        color_frames = frame_data['rgb_frame'].tolist()
        depth_frames = frame_data['depth_frame'].tolist()
        timestamps = frame_data['timestamp'].tolist()
        x_pos = frame_data['Robot odometry'].tolist()

        walking = False
        standing_still = False
        start_timestamp = 0
        end_timestamp = 0

        index = 0
        while True:
            frame_path = color_frames[index]
            depth_path = depth_frames[index]
            frame_path = check_os_windows(frame_path)
            depth_path = check_os_windows(depth_path)

            frame = cv2.imread(frame_path)
            depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            # if values from the depth map are greater than 5000, make them 0
            depth_map[depth_map > 6000] = 0

            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_HOT)
            added_images = cv2.addWeighted(frame, 0.7, depth_colormap, 0.3, 0)

            cv2.imshow('Frame viewer', added_images)
            key = cv2.waitKey(0) & 0xFF
            if key == 27: #ESC key
                break

            if key == ord('w'):
                if not walking and not standing_still:
                    walking = True
                    start_timestamp = timestamps[index]
                    print(f'Start walking at {start_timestamp}')
                    continue

                if walking:
                    walking = False
                    end_timestamp = timestamps[index]
                    print(f'Person Walking is still walking at {end_timestamp}')

                    if start_timestamp >= end_timestamp:
                        print('End timestamp must be greater than start timestamp.')
                        continue
                    else:
                        self.action_counter += 1
                        self.save_frames(start_timestamp, end_timestamp, 'walking', 0, self.action_counter)

                if standing_still:
                    standing_still = False
                    end_timestamp = timestamps[index]
                    print(f'Person Standing Still started walking at {end_timestamp}')

                    if start_timestamp >= end_timestamp:
                        print('End timestamp must be greater than start timestamp.')
                        continue
                    else:
                        self.action_counter += 1
                        self.save_frames(start_timestamp, end_timestamp, 'standing_still', 1, self.action_counter)

            if key == ord('s'):
                if not standing_still and not walking:
                    standing_still = True
                    start_timestamp = timestamps[index]
                    print(f'Start standing still at {start_timestamp}')
                    continue

                if standing_still:
                    standing_still = False
                    end_timestamp = timestamps[index]
                    print(f'Person Standing Still is still standing still at {end_timestamp}')

                    if start_timestamp >= end_timestamp:
                        print('End timestamp must be greater than start timestamp.')
                        continue
                    else:
                        self.action_counter += 1
                        self.save_frames(start_timestamp, end_timestamp, 'standing_still', 0, self.action_counter)

                if walking:
                    walking = False
                    end_timestamp = timestamps[index]
                    print(f'Person Walking is standing still at {end_timestamp}')

                    if start_timestamp >= end_timestamp:
                        print('End timestamp must be greater than start timestamp.')
                        continue
                    else:
                        self.action_counter += 1
                        self.save_frames(start_timestamp, end_timestamp, 'walking', 1, self.action_counter)

            if key == ord('a'):
                if index == 0:
                    print('Already at the first frame.')
                    continue
                else:
                    index -= 1
                    continue

            if key == ord('d'):
                if index == len(color_frames) - 1:
                    print('Already at the last frame.')
                    continue
                else:
                    index += 1
                    continue

            if key == ord('q'):
                break

        cv2.destroyAllWindows()

    def save_frames(self, start_timestamp, end_timestamp, action, intent, action_counter):
        """
        Save the frames from the start timestamp to the end timestamp in a folder.

        Args:
            start_timestamp (str): Start timestamp
            end_timestamp (str): End timestamp
            action (str): Action to save the frames
        """
        print(f"Saving frames from {start_timestamp} to {end_timestamp} for action {action} number {action_counter}")

        if self.check_intention:
            if intent == 0:
                class_name = f"{action}_no_intent"
            elif intent == 1:
                class_name = f"{action}_with_intent"
            action_folder = self.output_folder / class_name / f"{action}_{action_counter}"
        else:
            action_folder = self.output_folder / f"{action}_{action_counter}"

        action_folder.mkdir(parents=True, exist_ok=True)

        frame_data = self.check_correct_frames_data()
        frames_to_save = frame_data[(frame_data['timestamp'] >= start_timestamp) & (frame_data['timestamp'] <= end_timestamp)].copy()

        # Explicitly cast the 'action' column to string dtype
        frames_to_save.loc['action'] = frames_to_save['action'].astype  (str)
        frames_to_save.loc[:, 'action'] = action # or any default value

        if 'intent' not in frames_to_save.columns:
            frames_to_save.loc[:,'intent'] = 0 # or any default value

        # Explicitly cast the 'intent' column to integer dtype
        frames_to_save.loc['intent'] = frames_to_save['intent'].astype(int)
        frames_to_save.loc[:, 'intent'] = intent

        csv_output_path = action_folder / f"{action}_{action_counter}.csv"
        frames_to_save.to_csv(csv_output_path, index=False)

        for _, row in frames_to_save.iterrows():
            color_frame_path = check_os_windows(row['rgb_frame'])
            depth_frame_path = check_os_windows(row['depth_frame'])
            if color_frame_path.exists() and depth_frame_path.exists():
                shutil.copy(color_frame_path, action_folder / color_frame_path.name)
                shutil.copy(depth_frame_path, action_folder / depth_frame_path.name)
            else:
                print(f"Frame {row['rgb_frame']} does not exist.")



def main():
    frames_data = Path('/media/felipezero/T7 Shield/DATA/thesis/Videos/video_02/navigation_data.csv')
    output_folder = Path('/media/felipezero/T7 Shield/DATA/thesis/intent_prediction_dataset/classes_02/')

    HumanActionClassCreation(frames_data=frames_data, output_folder=output_folder, check_intention = True)


if __name__ == '__main__':
    main()