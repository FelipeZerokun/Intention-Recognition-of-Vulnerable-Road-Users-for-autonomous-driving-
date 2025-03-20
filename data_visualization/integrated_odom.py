from data_visualization import odometry_visualization, navigation_visualization
import matplotlib.pyplot as plt

def main():
    data_path = '../../DATA/pedestrians/navigation_data.csv'
    odometry_visualization.OdometryPlotter(data_path)
    navigation_visualization.NavigationPlotter(data_path)

    odom_points = odometry_visualization.cleaned_odometry_values
    nav_points = navigation_visualization.get_position('NED')

    plt.figure(figsize=(8, 6))
    plt.plot(odom_points[:, 0], odom_points[:, 1], label='Odometry')
    plt.plot(nav_points[:, 0], nav_points[:, 1], label='Navigation')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Odometry vs Navigation')
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()