import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt

from scipy.stats import zscore


class OdometryPlotter:

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

        # Extract the timestamp and odometry (assuming columns exist)
        self.timestamps = self.df["timestamp"].values
        self.odometry_data = self.df["Robot odometry"]
        self.odometry_values = self.extract_odometry_values(self.odometry_data)
        self.cleaned_odometry_values = self.fix_wrong_values(self.odometry_values)

    def extract_odometry_values(self, odometry_data):
        values = list(map(lambda s: list(map(float, re.findall(r'-?\d+\.\d+', s))), odometry_data))
        # Ensure all elements have the same length (e.g., 3 values per odometry entry)
        values = [v for v in values if len(v) == 3]
        return values

    def fix_wrong_values(self, odometry_values, threshold=500):
        fixed_values = []
        for i, (x, y, z) in enumerate(odometry_values):
            if i == 0:
                fixed_values.append([x, y, z])
                continue

            prev_x, prev_y, prev_z = fixed_values[-1]
            if abs(x - prev_x) > threshold or abs(y - prev_y) > threshold:
                avg_x = (prev_x + x) / 2
                avg_y = (prev_y + y) / 2
                fixed_values.append([avg_x, avg_y, z])
            else:
                fixed_values.append([x, y, z])

        return fixed_values

    def plot_odometry(self):
        # Extract X, Y positions from cleaned odometry
        x_positions = np.array([odo[0] for odo in self.cleaned_odometry_values])
        y_positions = np.array([odo[1] for odo in self.cleaned_odometry_values])

        # Ignore first 10 values (optional)
        x_positions = x_positions[10:]
        y_positions = y_positions[10:]

        # Plot the trajectory point by point
        plt.figure(figsize=(8, 6))
        for i in range(len(x_positions)):
            if i > 0:
                prev_x, prev_y = x_positions[i - 1], y_positions[i - 1]
                if abs(x_positions[i] - prev_x) > 100 or abs(y_positions[i] - prev_y) > 100:
                    plt.plot(x_positions[i], y_positions[i], marker="o", color="r")  # Highlight odd values
                else:
                    plt.plot(x_positions[i], y_positions[i], marker="o", color="b")
            else:
                plt.plot(x_positions[i], y_positions[i], marker="o", color="b")

        # Mark the starting point
        plt.plot(x_positions[0], y_positions[0], marker="s", color="g", markersize=10, label="Start Point")

        # Mark the end point
        plt.plot(x_positions[-1], y_positions[-1], marker="X", color="r", markersize=10, label="End Point")

        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Robot Trajectory from Odometry Data")
        plt.legend()
        plt.grid()
        plt.show()

if __name__ == "__main__":
    odometry_plotter = OdometryPlotter("../../DATA/pedestrians/navigation_data.csv")
    odometry_plotter.plot_odometry()