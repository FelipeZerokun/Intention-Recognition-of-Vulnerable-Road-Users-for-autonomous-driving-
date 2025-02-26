import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

def trajectory_plotter(pedestrian_data_dir):
    pedestrian_data = pd.read_csv(pedestrian_data_dir)

    # Convert the 'position' column from string to list using json.loads
    pedestrian_data['position'] = pedestrian_data['position'].apply(json.loads)

    # Extract X, Y, and Z coordinates
    x_ped, y_ped, z_ped = zip(*pedestrian_data['position'])

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.plot(x_ped, z_ped, marker='o', linestyle='-', color='b', label="Pedestrian Trajectory")

    # Mark the robot's position
    plt.scatter(0, 0, color='r', s=200, label="Robot (Fixed)")

    # Annotate start and end positions
    plt.text(x_ped[0], z_ped[0], "Start", fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    plt.text(x_ped[-1], z_ped[-1], "End", fontsize=12, verticalalignment='bottom', horizontalalignment='left')

    # Set plot labels and style
    plt.xlabel("X Position (meters)")
    plt.ylabel("Z Position (meters)")
    plt.title("Pedestrian Trajectory Relative to Robot")
    plt.axhline(0, color='black', linewidth=1)  # Ground line
    plt.axvline(0, color='black', linewidth=1)  # Robot's position
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    pedestrian_data_dir = "../datasets/pedestrian_2.csv"
    trajectory_plotter(pedestrian_data_dir)


def trajectory_plotter(pedestrian_data_dir):
    pedestrian_data = pd.read_csv(pedestrian_data_dir)

    # Convert the 'position' column from string to list using json.loads
    pedestrian_data['position'] = pedestrian_data['position'].apply(json.loads)
    pedestrian_id = pedestrian_data['track_id'].iloc[0]

    # Extract X, Y, and Z coordinates
    x_ped, y_ped, z_ped = zip(*pedestrian_data['position'])

    # Calculate the actual Z coordinate (assuming z_ped is the linear distance)
    z_ped = np.sqrt(np.array(z_ped) ** 2 - np.array(x_ped) ** 2)

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.plot(x_ped, z_ped, marker='o', linestyle='-', color='b', label="Pedestrian Trajectory")

    # Mark the robot's position
    plt.scatter(0, 0, color='r', s=200, label="Robot (Fixed)")

    # Annotate start and end positions
    plt.text(x_ped[0], z_ped[0], "Start", fontsize=12, verticalalignment='bottom', horizontalalignment='right')
    plt.text(x_ped[-1], z_ped[-1], "End", fontsize=12, verticalalignment='bottom', horizontalalignment='left')

    # Draw an arrow pointing in the X direction of the robot
    plt.annotate('', xy=(1, 0), xytext=(0, 0),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))

    # Set plot labels and style
    plt.xlabel("X Position (meters)")
    plt.ylabel("Z Position (meters)")
    plt.title(f"Pedestrian {pedestrian_id} Trajectory Relative to Robot")
    plt.axhline(0, color='black', linewidth=1)  # Ground line
    plt.axvline(0, color='black', linewidth=1)  # Robot's position
    plt.legend()
    plt.grid()

    # Show the plot
    plt.show()


if __name__ == "__main__":
    pedestrian_data_dir = "../datasets/pedestrian_1.csv"
    trajectory_plotter(pedestrian_data_dir)