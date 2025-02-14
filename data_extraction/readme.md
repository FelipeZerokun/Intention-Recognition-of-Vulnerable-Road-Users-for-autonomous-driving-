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
