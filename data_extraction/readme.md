# Data extraction from rosbags

## Rosbags

Rosbags were created using Ros1. So, to extract the data from them, it is needed to use Ros Noetic.

I used a Docker container with Ros Noetic for this purpose and used python with the module Rosbag

## Extracting data

### Checking the topics

First, I check the topics inside the Rosbags. I note down the topics that I need for the pedestrian intent prediction.

from the Depth Camera:
- /camera/color/camera_info: Camera info for calibration.
- /camera/color/image_raw: Raw image from the camera.
- /camera/depth/image_rect_raw: Depth image from the camera.
- /camera/depth/color/points: Point cloud from the camera.

From the GPS module:
- /anavs/solution/pos: Global velocity of the robot using North, East and Down.
- /anavs/solution/vel: Estimated velocity of the robot.
- /anavs/solution/pos_llh: Global position of the robot using Longitude, Latitude and Height.
- /anavs/solution/ecef: Global position of the robot using Earth-Centered, Earth-Fixed coordinates.
- /anavs/solution/att: Attitude of the robot.

From Robot:
- /RosAria/pose: Pose of the robot.

### Extracting the data
There are several functions to extract data depending on the type of data. The functions are:
- get_camera_info: Extracts the camera info from the topic /camera/color/camera_info.
- get_odometry: Extracts the odometry data from the topic /RosAria/pose.
- get_llh_position: Extracts the GPS position data (Latitude, Longitude, Height) from the topic /anavs/solution/pos_llh.
- get_NED_position: Extracts the GPS position data (North, East, Down) from the topic /anavs/solution/pos.
- get_ECEF_position: Extracts the GPS position data (Earth-Centered, Earth-Fixed) from the topic /anavs/solution/ecef.
- get_velocity: Extracts the GPS velocity data from the topic /anavs/solution/vel.
- get_color_image: Extracts the color image from the topic /camera/color/image_raw.
- get_depth_image: Extracts the depth image from the topic /camera/depth/image_rect_raw.
  - In case the Depth image needs to be cropped to match the color image, there is a parameters fix_frame.
- convert_depth2image: Function to crop and resize images.
- visualize_data: Function to visualize both Color and Depth images
- get_pointcloud_map: Extracts the data from the topic /camera/depth/color/points and converts it into a PointCloud file.
- pointcloud_to_image: Function to convert the PointCloud data into an image.

### Storing the data
Depending on the required data, several functions can extract and save the data.
- extract_all_data: Extracts data from all the Rostopics and saves them in the output folder.
  - Data is synchronized using the timestamps of the messages.
  - Odometry is used as the reference for the synchronization.
  - PointCloud data can be saved as a PCD file or as a PointCloud2 message.
  - Color and Depth images are saved as PNG files.
  - Navigation data is saved as a CSV file.
- extract_stereo_data: Extracts only data from the Depth camera.
- extract_video: Extracts color images and converts them into a video format.
  - The video is saved in the output folder.
  - The video is encoded using X264 codec for MP4 format
- 