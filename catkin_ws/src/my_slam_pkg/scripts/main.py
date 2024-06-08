from ArucoSLAM import *

# Function to run the SLAM process
def run_slam(rosbag_file, nr_map, file, slam_variables, occupancy_map):
    rosbag_process = None
    
    # Check if the rosbag file is for microsimulation
    if rosbag_file == 'microsim':
        rosbag_process = subprocess.Popen(['python3', '../micro_simulation/main.py'])
        return
              
    # If not microsimulation, handle different rosbag file scenarios
    if rosbag_file != 'microsim':
        try:
            # Launch teleoperation for live mode
            if rosbag_file == 'live':
                rosbag_process = subprocess.Popen(['roslaunch', 'turtlebot3_teleop', 'turtlebot3_teleop_key.launch'])
            else:
                # Play the rosbag file
                rosbag_process = subprocess.Popen(['rosbag', 'play', rosbag_file])
                # Launch RViz for visualization
                rviz = subprocess.Popen(['roslaunch', 'turtlebot3_gazebo', 'turtlebot3_gazebo_rviz.launch'])
                
                # Check if the rosbag file exists
                if not os.path.isfile(rosbag_file):
                    print(f"ERROR: The file {rosbag_file} does not exist.")
                    exit(1)
                
            # Get the duration of the rosbag file
            rosbag_time = get_rosbag_duration(rosbag_file)
            # Initialize SLAM
            slam = ArucoSLAM(rosbag_time, slam_variables, occupancy_map)
            # Run the SLAM process
            slam.run()
            # Compute and print metrics
            if nr_map == 5:
                distance = compute_metrics(slam, nr_map, file)
                print(f"The trajectory error for {rosbag_file} is: {distance}")
            else:
                ate, rpe, sse_landmarks = compute_metrics(slam, nr_map, file)
                print(f"Metrics for {rosbag_file}:")
                print(f"ATE: {ate}, RPE: {rpe}, SSE Landmarks: {sse_landmarks}")
        finally:
            # Terminate the rosbag process
            if rosbag_process:
                rosbag_process.terminate()

# Function to set SLAM variables
def variables():
    occupancy_map = True

    window_size_pixel = 900  # Pixel size of window
    size_window = 10  # Size of window in meters
    OG_map_options = (20, 20, 0.1)  # Width meters, height meters, resolution meters per cell
    number_particles = 25
    Q_init = np.diag([0.1, 0.1])
    Q_update = np.diag([0.7, 0.7])
    alphas = [0.00008, 0.00008, 0.00001, 0.00001]
    tuning_option = [Q_init, Q_update, alphas]
    
    return (window_size_pixel, size_window, OG_map_options, number_particles, tuning_option), occupancy_map

# Main execution block
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select a rosbag to run.")
    parser.add_argument('rosbag', help="The rosbag file to run.")
    args = parser.parse_args()
    nr_map = None
    file = args.rosbag
    slam_variables, occupancy_map = variables()
    
    # Check if the provided argument is a rosbag file
    if args.rosbag.endswith('.bag'):
        rosbag_file = f"../rosbag/{args.rosbag}"
        match = re.search(r"\d", args.rosbag)  # Search for the first digit of the name to determine the map number
        nr_map = int(match.group(0))
        if match:
            print("Map found:", nr_map)
        else:
            print("No map found")
    elif args.rosbag == 'live':
        rosbag_file = 'live'
    elif args.rosbag == 'microsim':
        rosbag_file = 'microsim'
    else:
        print("Invalid choice. Exiting.")
        exit(1)
    
    # Run the SLAM process
    run_slam(rosbag_file, nr_map, file, slam_variables, occupancy_map)
