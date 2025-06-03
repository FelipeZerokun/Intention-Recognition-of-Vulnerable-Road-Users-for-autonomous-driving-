# Action Recognition and Intent prediction Dataset Manager Module

The `Dataset Manager` module is designed to handle datasets for human action and intent recognition. It provides tools for processing video frames, extracting data, and organizing it into classes for action and intent prediction. This module is particularly useful for creating labeled datasets for machine learning models.

## Features

- **Camera Information Extraction**: Reads camera parameters such as image size and intrinsic matrix.
- **Frame Data Validation**: Ensures the dataset contains the required columns for processing.
- **Action and Intent Class Creation**: Allows manual labeling of actions and intents by reviewing video frames.
- **Pedestrian Tracking**: Tracks pedestrians in video frames using the DeepSort tracker.
- **RGB and Depth Frame Combination**: Combines RGB and depth frames for visualization.
- **Frame Saving**: Saves labeled frames and metadata into organized folders.

## Class and Functions

### `DatasetManager`

#### **Attributes**
- `data_path (str)`: Path to the dataset directory.
- `video (str)`: Name of the folder that contains the data from the rosbags (example: video_02/test_10, etc...)
- `output_folder (str)`: Path to the output folder for saving processed data.


#### Usage

1. **Initialize the Dataset Manager**:
   ```python
   dataset_manager = DatasetManager(data_path="path/to/data", video="video_name", output_folder="path/to/output")

2. **Run function to create classes**:
- For action recognition:
    ```python
    dataset_manager.create_classes_for_action_recognition()
- For Intent prediction:
    ```python
    dataset_manager.create_classes_for_intent_prediction()

#### **Methods**

1. **`__init__(data_path: str, video: str, output_folder: str)`**
   - Initializes the `DatasetManager` with dataset paths and sets up the environment.

2. **`get_camera_info()`**
   - Extracts camera information such as image size and intrinsic matrix.
   - **Returns**: Tuple containing image size and intrinsic matrix.

3. **`check_correct_frames_data()`**
   - Validates the structure of the frames data CSV file.
   - **Raises**: `ValueError` if required columns are missing.
   - **Returns**: Validated `pandas.DataFrame`.

4. **`create_classes_for_action_recognition()`**
   - Allows manual creation of classes for action recognition by labeling video frames.

5. **`create_classes_for_intent_prediction()`**
   - Allows manual creation of classes for intent prediction by labeling video frames.

6. **`action_analysis(frame_data, start_timestamp, end_timestamp, pedestrian_counter, check_intention=False)`**
   - Analyzes pedestrian actions and intents within a specified time range.
   - **Args**:
     - `frame_data (pd.DataFrame)`: Dataframe containing frame data.
     - `start_timestamp (int)`: Start timestamp for analysis.
     - `end_timestamp (int)`: End timestamp for analysis.
     - `pedestrian_counter (int)`: Counter for pedestrians in the dataset.
     - `check_intention (bool)`: Whether to check pedestrian intent.
   - **Returns**: Updated pedestrian counter.

7. **`track_pedestrians(frame: np.ndarray, detections: np.ndarray)`**
   - Tracks pedestrians in a video frame using the DeepSort tracker.
   - **Args**:
     - `frame (np.ndarray)`: The video frame.
     - `detections (np.ndarray)`: Detected objects in the frame.
   - **Returns**: List of tracked pedestrian objects.

8. **`combine_rgb_depth_frames(frame: np.ndarray, depth_map: np.ndarray, depth_threshold=10000)`**
   - Combines an RGB frame and a depth map into a single image for visualization.
   - **Args**:
     - `frame (np.ndarray)`: The RGB frame.
     - `depth_map (np.ndarray)`: The depth map.
     - `depth_threshold (int)`: Maximum depth value to consider.
   - **Returns**: Combined image.

9. **`save_frames(start_timestamp: str, end_timestamp: str, action: str, intent: int, pedestrian_counter: int)`**
   - Saves frames and metadata for a specific action and intent into a folder.
   - **Args**:
     - `start_timestamp (str)`: Start timestamp for saving frames.
     - `end_timestamp (str)`: End timestamp for saving frames.
     - `action (str)`: Action label for the frames.
     - `intent (int)`: Intent label for the frames.
     - `pedestrian_counter (int)`: Counter for pedestrians in the dataset.

    
### `Class Analysis`
Support Class to review a specific class
   
#### **Attributes**
- `class_dir (str)`: Path to the pedestrian directory (example: dataset/walking/pedestrian_01)
   
- The function will check the images in the folder and the data inside the CSV file.
    - If the number of images in the folder is different from the number of rows in the CSV file, it will print a message indicating the difference.
    - If the pedestrian ID in the folder is different from the ID on each image and in the CSV file, the function will rename the images and change the CSV file.
    - Each image will be shown in a new window. A prompt asking if the image is correct will appear. Pressing 'N' will remove the image from the directory.

#### Usage

1. **Initialize the Dataset Manager**:
   ```python
   class_analysis = ClassAnalysis(class_directory)

### `Frames Extraction`
To extract the frames from a video

#### **Attributes**
- `video_path (str)`: Path to the video file.
- `output_folder (str)`: Path to the output folder for saving extracted frames.

#### Usage

1. **Initialize the Dataset Manager**:
   ```python
   frame_extractor = VideoLabeling(videos_dir, output_dir)
   video_labeling.extract_frames_single(frames=number_of_frames_per_second)

### Navigation Data Modifier
