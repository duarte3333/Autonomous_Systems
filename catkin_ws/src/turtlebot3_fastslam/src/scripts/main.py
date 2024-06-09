#!/usr/bin/env python3

from ArucoSLAM import *
import rospy
import rospkg

# Function to run the SLAM process
def run_slam(rosbag_file, nr_map,file,slam_variables,occupancy_map):
    rosbag_process = None
    try:
            if not os.path.isfile(rosbag_file):
                print(f"ERROR: The file {rosbag_file} does not exist.")
                exit(1)
                
            rosbag_time = get_rosbag_duration(rosbag_file)
            slam = ArucoSLAM(rosbag_time,slam_variables,occupancy_map)
            slam.run()
            if nr_map == 5:
                distance = compute_metrics(slam, nr_map,file)
                print(f"The trajectory error for {rosbag_file} is: {distance}")
            else:
                ate, rpe, sse_landmarks = compute_metrics(slam, nr_map,file)
                print(f"Metrics for {rosbag_file}:")
                print(f"ATE: {ate}, RPE: {rpe}, SSE Landmarks: {sse_landmarks}")
    finally:
            if rosbag_process:
                rosbag_process.terminate()

# Function to set SLAM variables
def variables():
    occupancy_map=True

    window_size_pixel=900    #pixel size of window
    size_window = 10 # in meters
    OG_map_options=(20,20,0.1) #width meters, height meters, resolution meters per cell
    number_particles=25
    Q_init=np.diag([0.1,0.1])
    Q_update=np.diag([0.7,0.7])
    alphas=[0.00008,0.00008,0.00001,0.00001]
    tuning_option=[Q_init,Q_update,alphas]
    return (window_size_pixel, size_window,OG_map_options,number_particles,tuning_option), occupancy_map

# Main execution block
if __name__ == '__main__':
    rosPackage=True
    current_directory = os.getcwd()

    if rosPackage:
        file = rospy.get_param('~rosbag', 'Mapa4_2.bag')
        rospy.loginfo(f'Rosbag file: {file}')

        
    else:
        parser = argparse.ArgumentParser(description="Select a rosbag to run.")
        parser.add_argument('rosbag', help="The rosbag file to run.")
        args = parser.parse_args()
        file = args.rosbag

    nr_map=None
    slam_variables, occupancy_map =variables()
    if file.endswith('.bag'):


        rospack = rospkg.RosPack()
        path=rospack.get_path('turtlebot3_fastslam')

        rosbag_file = f"{path}/src/rosbag/{file}"
        match = re.search(r"\d", file) #this searches for the first digit of the name
        nr_map=int(match.group(0))
        if match:
            print("Map found:", nr_map)
        else:
            print("No map found")

    else:
        print("Invalid choice. Exiting.")
        exit(1)

    # Run the SLAM process
    run_slam(rosbag_file, nr_map,file, slam_variables,occupancy_map)

    