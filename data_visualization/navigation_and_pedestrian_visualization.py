import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import cv2

from project_utils.project_utils import estimate_pedestrian_distance, estimate_pedestrian_position

# Load Navigation Data
file_path = "../../DATA/video_01/navigation_data.csv"  # Update this with your CSV file
df = pd.read_csv(file_path)


# Function to parse NED position from string
def parse_ned_string(ned_str):
    ned_list = ast.literal_eval(ned_str)  # Convert string to list
    return float(ned_list[0]), float(ned_list[1])  # Extract X (North), Y (East)


# Extract Robot's Path (Skipping first 10 values due to incorrect data)
ned_positions = df["Robot NED position"].apply(parse_ned_string)[10:]
robot_north, robot_east = zip(*ned_positions)  # Unpack into separate lists


# Function to estimate pedestrian position in robot frame
def estimate_pedestrian_position(bbox, depth_image):
    fx, fy = 525.0, 525.0  # Camera focal length (update if different)
    cx, cy = 320.0, 240.0  # Camera principal point (update if different)

    x_min, y_min, x_max, y_max = bbox
    u = int((x_min + x_max) / 2)
    v = int((y_min + y_max) / 2)

    Z = depth_image[v, u] / 1000.0  # Convert mm to meters
    if Z == 0:
        return None  # Invalid depth

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    return X, Y  # Position relative to the robot


# Example: Assume we have detected pedestrians at each timestamp
# Here, we use dummy pedestrian bounding boxes for demonstration
pedestrian_positions = []
depth_image = cv2.imread("path/to/depth_image.png", cv2.IMREAD_UNCHANGED)  # Load depth image

for _ in range(len(ned_positions)):  # Simulate one detection per timestamp
    bbox = (150, 100, 250, 300)  # Example pedestrian bbox (replace with actual detections)
    pedestrian_pos = estimate_pedestrian_position(bbox, depth_image)

    if pedestrian_pos:
        # Convert pedestrian position to global NED frame
        ped_x = robot_north[_] + pedestrian_pos[0]  # North
        ped_y = robot_east[_] + pedestrian_pos[1]  # East
        pedestrian_positions.append((ped_x, ped_y))
    else:
        pedestrian_positions.append(None)  # No detection

# Plot Robot and Pedestrian Paths for 50 timestamps
plt.figure(figsize=(8, 6))
plt.plot(robot_east[:50], robot_north[:50], marker="o", linestyle="-", color='b', label="Robot Path")

for i, ped_pos in enumerate(pedestrian_positions[:50]):
    if ped_pos:
        plt.scatter(ped_pos[1], ped_pos[0], color='r', marker='x', label="Pedestrian" if i == 0 else "")

plt.xlabel("East (m)")
plt.ylabel("North (m)")
plt.title("Robot and Pedestrian Trajectories (Last 50 timestamps)")
plt.legend()
plt.grid()
plt.show()
