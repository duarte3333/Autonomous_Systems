#!/usr/bin/env python3
import rospy
import subprocess
import argparse
import os
import cv2
import cv2.aruco as aruco
import numpy as np
import threading
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from FastSlam import FastSlam
import re
import pygame
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from metrics import compute_metrics
from callbacks import odom_callback, lidar_callback
from img_callback import image_callback
from aux_slam import get_rosbag_duration, cart2pol


class ArucoSLAM:
    def odom_callback_wrapper(self, data):
        odom_callback(self, data)

    def image_callback_wrapper(self, data):
        image_callback(self, data)

    def lidar_callback_wrapper(self,data):
        lidar_callback(self,data)

    def __init__(self, rosbag_time):
        self.image_callback = image_callback
        self.odom_callback = odom_callback
        self.k = rosbag_time + 5
        self.calibrate_camera()
        self.lock = threading.Lock()
        
        rospy.loginfo('ArucoSLAM Node Started')
        rospy.init_node('aruco_slam') # Initialize the ROS node
        Occu_grid_pub = rospy.Publisher('/occupancy_grid', OccupancyGrid, queue_size=10)
        self.create_slam(Occu_grid_pub)

        #subscribe node
        self.image_sub = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.image_callback_wrapper)
        self.current_aruco = []  

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback_wrapper)
        self.odom =[0,0,0]
        self.bridge = CvBridge() # Initialize the CvBridge object

        rospy.Subscriber('/scan', LaserScan, self.lidar_callback_wrapper)

        
        try: #The tag of our aruco dictionary is 4X4_100
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
            print(self.aruco_dict)
        except:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250) 
            print("2: ", self.aruco_dict)

        #Do not change the tag used (cv2.aruco.DICT_4X4_100), only if we change the ArUcos we're using (and thus their tags) 
        try:
            self.parameters = aruco.DetectorParameters_create()
        except:
            self.parameters =  cv2.aruco.DetectorParameters()
        self.dict = {}
        self.map = {}


    def create_slam(self,Occu_grid_pub):
        window_size_pixel=900    #tamanho da janela
        sample_rate=5  #sample rate (Hz)
        size_m = 10#float(input('What should be the size of the map? n x n (in meters). n is: '))
        central_bar_width=10
        turtlebot_L=0.287
        OG_map_options=(20,20,0.1) #width meters, height meters, resolution meters per cell
        number_particles=1
        self.my_slam = FastSlam(True, window_size_pixel, sample_rate, size_m, central_bar_width,OG_map_options,Occu_grid_pub, turtlebot_L,number_particles)
        self.count = 0

    def calibrate_camera(self):
        dist =[0.1639958233797625, -0.271840030972792, 0.001055841660100477, -0.00166555973740089, 0.0]
        K = [322.0704122808738, 0.0, 199.2680620421962, 0.0, 320.8673986158544, 155.2533082600705, 0.0, 0.0, 1.0]
        P = [329.2483825683594, 0.0, 198.4101510452074, 0.0, 0.0, 329.1044006347656, 155.5057121208347, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        #K[1,2] , K[2,2], Tx, Ty
        K[2] = 320 ; K[5] = 240 ; P[2] = 320 ; P[6] = 240
        
        mtx = np.array(K).reshape(3, 3)
        dist = np.array(dist).reshape(1,5)
        self.camera_matrix = mtx.astype(float)
        self.dist_coeffs = dist.astype(float)
        
    def run(self):
        start_time = rospy.Time.now() 
        while not rospy.is_shutdown():  # Continue running until ROS node is shutdown
            # Check if the rosbag playback has finished
            if self.rosbag_finished(start_time, self.k):  # Implement this function to check if the rosbag playback has finished
                rospy.loginfo("Rosbag playback finished. Shutting down...")
                rospy.signal_shutdown("Rosbag playback finished")  # Shutdown ROS node
                break  # Exit the loop
            rospy.sleep(0.1) # Process ROS callbacks once
        cv2.destroyAllWindows()  # Close OpenCV windows when node exits
        pygame.display.quit()
        pygame.quit()
    
    def rosbag_finished(self, start_time, duration):
        # Calculate the expected end time based on the start time and duration of rosbag playback
        end_time = start_time + rospy.Duration.from_sec(duration)
        # Check if the current ROS time exceeds the expected end time
        if rospy.Time.now() > end_time:
            return True  # Playback finished
        else:
            return False  # Playback ongoing

    def get_trajectory(self):
        return self.my_slam.get_best_trajectory()

def run_slam(rosbag_file, nr_map):
    rosbag_process = None
    if (rosbag_file == 'microsim'):
        rosbag_process = subprocess.Popen(['python3', '../micro_simulation/main.py'])
        return
              
    if rosbag_file != 'microsim':
        try:
            if (rosbag_file == 'live'):
                rosbag_process = subprocess.Popen(['roslaunch', 'turtlebot3_teleop', 'turtlebot3_teleop_key.launch'])
            else:
                rosbag_process = subprocess.Popen(['rosbag', 'play', rosbag_file])
                if not os.path.isfile(rosbag_file):
                    print(f"ERROR: The file {rosbag_file} does not exist.")
                    exit(1)
                
            rosbag_time = get_rosbag_duration(rosbag_file)
            slam = ArucoSLAM(rosbag_time)
            gt = 0 
            slam.run()
            ate, rpe, mse_landmarks = compute_metrics(slam, gt, nr_map)
            print(f"Metrics for {rosbag_file}:")
            print(f"ATE: {ate}, RPE: {rpe}, MSE Landmarks: {mse_landmarks}")
        finally:
            if rosbag_process:
                rosbag_process.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select a rosbag to run.")
    parser.add_argument('rosbag', help="The rosbag file to run.")
    args = parser.parse_args()
    nr_map=None


    if args.rosbag.endswith('.bag'):
        #print("entrei", args.rosbag)
        rosbag_file = f"../rosbag/{args.rosbag}"
        match = re.search(r"\d", args.rosbag) #this searches for the first digit of the name
        nr_map=int(match.group(0))
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
    run_slam(rosbag_file, nr_map)

    