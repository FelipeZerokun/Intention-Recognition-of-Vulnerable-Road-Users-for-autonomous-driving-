import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ast  # To safely parse the string as a list
from matplotlib.ticker import FormatStrFormatter

class NavigationPlotter:

    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)

    def parse_position_string(self, position_str):
        position_list = ast.literal_eval(position_str)  # Convert string to list
        return tuple(map(float, position_list))  # Convert all elements to float and return as tuple

    def get_position(self, gps_mode: str):
        if gps_mode == 'NED':
            ned_data = self.df["Robot NED position"].apply(self.parse_position_string)
            north_positions, east_positions, _ = zip(*ned_data)
            # Ignore the first 10 values
            north_positions = north_positions[10:]
            east_positions = east_positions[10:]
            return north_positions, east_positions

        elif gps_mode == 'ECEF':
            ecef_data = self.df["Robot ECEF position"].apply(self.parse_position_string)
            x_positions, y_positions, z_positions = zip(*ecef_data)
            # Ignore the first 10 values
            x_positions = x_positions[10:]
            y_positions = y_positions[10:]
            z_positions = z_positions[10:]
            return x_positions, y_positions, z_positions

        elif gps_mode == 'LLH':
            llh_data = self.df["Robot Global position"].apply(self.parse_position_string)
            latitudes, longitudes, heights = zip(*llh_data)
            # Ignore the first 10 values
            latitudes = latitudes[10:]
            longitudes = longitudes[10:]
            heights = heights[10:]

            return latitudes, longitudes, heights

    def plot_navigation(self, mode = 'ALL'):

        if mode == 'NED':
            # Plot NED trajectory
            north_positions, east_positions = self.get_position('NED')
            plt.figure(figsize=(10, 6))
            plt.plot(east_positions, north_positions, marker="o", linestyle="-", markersize=2, label="NED Path",
                     color='r')
            plt.xlabel("East (m)")
            plt.ylabel("North (m)")
            plt.title("Robot Navigation Path (NED Coordinates)")
            plt.legend()
            plt.grid()
            plt.show()

        elif mode == 'ECEF_3D':
            # 3D Plot of ECEF trajectory
            x_positions, y_positions, z_positions = self.get_position('ECEF')
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(x_positions, y_positions, z_positions, marker="o", linestyle="-", markersize=2, label="ECEF Path",
                    color='g')

            ax.set_xlabel("ECEF X (m)")
            ax.set_ylabel("ECEF Y (m)")
            ax.set_zlabel("ECEF Z (m)")
            ax.set_title("Robot Navigation Path (ECEF Coordinates)")
            ax.legend()
            plt.show()

        elif mode == 'ECEF_2D':
            # 2D Plot of ECEF trajectory
            x_positions, y_positions, _ = self.get_position('ECEF')
            plt.figure(figsize=(10, 6))
            plt.plot(x_positions, y_positions, marker="o", linestyle="-", markersize=2, label="ECEF Path", color='g')
            plt.xlabel("ECEF X (m)")
            plt.ylabel("ECEF Y (m)")
            plt.title("Robot Navigation Path (ECEF Coordinates)")
            plt.legend()
            plt.grid()
            plt.show()


        elif mode == 'LLH_3D':

            # Plot the GPS trajectory

            latitudes, longitudes, heights = self.get_position('LLH')

            fig = plt.figure(figsize=(10, 7))

            ax = fig.add_subplot(111, projection='3d')

            ax.plot(longitudes, latitudes, heights, marker="o", linestyle="-", markersize=2, label="GPS Path",
                    color='b')

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_zlabel("Height (m)")
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax.set_title("Robot Navigation Path (LLH Coordinates 3D)")
            ax.legend()
            plt.show()


        elif mode == 'LLH_2D':

            # Plot the GPS trajectory

            latitudes, longitudes, _ = self.get_position('LLH')

            plt.figure(figsize=(10, 6))

            plt.plot(longitudes, latitudes, marker="o", linestyle="-", markersize=2, label="GPS Path", color='b')
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.title("Robot Navigation Path (LLH)")

            plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            plt.legend()
            plt.grid()
            plt.show()

        elif mode == 'ALL_2D':
            # plots the NED, ECEF and LLH trajectories in 2D
            north_positions, east_positions = self.get_position('NED')
            x_positions, y_positions, _ = self.get_position('ECEF')
            latitudes, longitudes, _ = self.get_position('LLH')

            fig = plt.figure(figsize=(15, 10))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)

            # NED
            ax1.plot(east_positions, north_positions, marker="o", linestyle="-", markersize=2, label="NED Path",
                     color='r')
            ax1.set_xlabel("East (m)")
            ax1.set_ylabel("North (m)")
            ax1.set_title("Robot Navigation Path (NED Coordinates)")
            ax1.legend()
            ax1.grid()

            # ECEF
            ax2.plot(x_positions, y_positions, marker="o", linestyle="-", markersize=2, label="ECEF Path",
                        color='g')
            ax2.set_xlabel("ECEF X (m)")
            ax2.set_ylabel("ECEF Y (m)")
            ax2.set_title("Robot Navigation Path (ECEF Coordinates)")
            ax2.legend()
            ax2.grid()

            # LLH 2D
            ax3.plot(longitudes, latitudes, marker="o", linestyle="-", markersize=2, label="GPS Path", color='b')
            ax3.set_xlabel("Longitude")
            ax3.set_ylabel("Latitude")
            ax3.set_title("Robot Navigation Path (GPS)")
            ax3.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax3.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax3.legend()
            ax3.grid()
            plt.show()


        elif mode == 'ALL':

            # Plot all the trajectories in one figure.
            # NED top left, LLH bottom left, ECEF right, ECEF 3D bottom right
            north_positions, east_positions = self.get_position('NED')
            x_positions, y_positions, z_positions = self.get_position('ECEF')
            latitudes, longitudes, heights = self.get_position('LLH')

            fig = plt.figure(figsize=(15, 10))
            ax1 = fig.add_subplot(221)
            ax2 = fig.add_subplot(222)
            ax3 = fig.add_subplot(223)
            ax4 = fig.add_subplot(224, projection='3d')

            # NED
            ax1.plot(east_positions, north_positions, marker="o", linestyle="-", markersize=2, label="NED Path",
                     color='r')
            ax1.set_xlabel("East (m)")
            ax1.set_ylabel("North (m)")
            ax1.set_title("Robot Navigation Path (NED Coordinates)")
            ax1.legend()
            ax1.grid()

            # LLH 2D
            ax2.plot(longitudes, latitudes, marker="o", linestyle="-", markersize=2, label="GPS Path", color='b')
            ax2.set_xlabel("Longitude")
            ax2.set_ylabel("Latitude")
            ax2.set_title("Robot Navigation Path (LLH Coordinates)")
            ax2.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
            ax2.legend()
            ax2.grid()

            # ECEF
            ax3.plot(x_positions, y_positions, marker="o", linestyle="-", markersize=2, label="ECEF Path",
                        color='g')
            ax3.set_xlabel("ECEF X (m)")
            ax3.set_ylabel("ECEF Y (m)")
            ax3.set_title("Robot Navigation Path (ECEF Coordinates)")
            ax3.legend()
            ax3.grid()

            # ECEF 3D
            ax4.plot(x_positions, y_positions, z_positions, marker="o", linestyle="-", markersize=2, label="ECEF Path",
                    color='g')
            ax4.set_xlabel("ECEF X (m)")
            ax4.set_ylabel("ECEF Y (m)")
            ax4.set_zlabel("ECEF Z (m)")
            ax4.set_title("Robot Navigation Path (ECEF Coordinates 3D)")
            ax4.legend()
            ax4.grid()
            plt.show()


def main():
    file_path = "../../DATA/video_01/navigation_data.csv"
    navigation_plotter = NavigationPlotter(file_path)
    navigation_plotter.plot_navigation(mode='NED')
    navigation_plotter.plot_navigation(mode='ECEF_2D')
    navigation_plotter.plot_navigation(mode='LLH_2D')
    navigation_plotter.plot_navigation(mode='ALL_2D')
    navigation_plotter.plot_navigation(mode='ECEF_3D')
    navigation_plotter.plot_navigation(mode='LLH_3D')
    navigation_plotter.plot_navigation(mode='ALL')

if __name__ == '__main__':
    main()