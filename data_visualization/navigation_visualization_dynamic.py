import pandas as pd
import folium
import ast  # To parse string list format

# Load the CSV file
file_path = "../../DATA/video_01/navigation_data.csv"  # Change this to your actual file path
df = pd.read_csv(file_path)

# Function to parse LLH GPS data from string
def parse_llh_string(llh_str):
    llh_list = ast.literal_eval(llh_str)  # Convert string to list
    return float(llh_list[0]), float(llh_list[1])  # Extract Latitude, Longitude

# Extract LLH data
gps_data = df["Robot Global position"].apply(parse_llh_string)
latitudes, longitudes = zip(*gps_data)  # Unpack into separate lists

# Skip the first 10 incorrect values
latitudes = latitudes[10:]
longitudes = longitudes[10:]

# Initialize a Folium map centered at the starting position
start_location = [latitudes[0], longitudes[0]]
m = folium.Map(location=start_location, zoom_start=17)  # Adjust zoom as needed

# Add a PolyLine to show the robot's path
path = list(zip(latitudes, longitudes))
folium.PolyLine(path, color="blue", weight=3, opacity=0.7).add_to(m)

# Add start and end markers
folium.Marker(location=path[0], popup="Start", icon=folium.Icon(color="green")).add_to(m)
folium.Marker(location=path[-1], popup="End", icon=folium.Icon(color="red")).add_to(m)

# Save and display the map
map_filename = "robot_navigation_map.html"
m.save(map_filename)
print(f"Map saved as {map_filename}. Open it in a browser to view the interactive map.")
