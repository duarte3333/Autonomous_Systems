#!/usr/bin/env python3

# Import necessary libraries
import rospy
import subprocess
import argparse
import os
import cv2
import cv2.aruco as aruco
import numpy as np
import threading
import tf
from geometry_msgs.msg import TransformStamped
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
from visualization_msgs.msg import Marker, MarkerArray

class ArucoSLAM:
    def __init__(self, rosbag_time, slam_variables, occupancy_map):
        # Initialize instance variables and set up ROS node
        self.image_callback = image_callback
        self.odom_callback = odom_callback
        self.k = rosbag_time + 5
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.calibrate_camera()  # Camera calibration
        self.lock = threading.Lock()
        
        rospy.loginfo('ArucoSLAM Node Started')
        rospy.init_node('aruco_slam')  # Initialize the ROS node
        Occu_grid_pub = rospy.Publisher('/occupancy_grid', OccupancyGrid, queue_size=10)
        self.create_slam(Occu_grid_pub, slam_variables)  # Create SLAM object

        # Subscribe to relevant ROS topics
        self.image_sub = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.image_callback_wrapper)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback_wrapper)
        self.landmark_pub = rospy.Publisher('/landmarks', MarkerArray, queue_size=10)
        self.current_aruco = []  
        self.odom = [0,0,0]
        self.bridge = CvBridge()  # Initialize the CvBridge object

        if occupancy_map:
            rospy.Subscriber('/scan', LaserScan, self.lidar_callback_wrapper)

        # Load ArUco dictionary
        try:
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        except:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250) 

        # Create ArUco detector parameters
        try:
            self.parameters = aruco.DetectorParameters_create()
        except:  # For different versions of OpenCV
            self.parameters = cv2.aruco.DetectorParameters()
        self.dict = {}
        self.map = {}

    # Wrapper for odometry callback
    def odom_callback_wrapper(self, data):
        odom_callback(self, data)

    # Wrapper for image callback
    def image_callback_wrapper(self, data):
        image_callback(self, data)

    # Wrapper for lidar callback
    def lidar_callback_wrapper(self, data):
        lidar_callback(self, data)

    # Publish transformation frames
    def publish_tf(self):
        # Broadcast map to odom
        br = tf.TransformBroadcaster()
        br.sendTransform(
            (0.0, 0.0, 0.0),
            tf.transformations.quaternion_from_euler(0, 0, 0),
            rospy.Time.now(),
            "odom",
            "map"
        )

        # Get best particle and broadcast odom to base_link
        best_particle = self.my_slam.get_best_particle()
        x, y, theta = best_particle.pose  # Access the pose property correctly
        quaternion = tf.transformations.quaternion_from_euler(0, 0, theta - (np.pi))
        br.sendTransform(
            (-x, y, 0),
            quaternion,
            rospy.Time.now(),
            "base_link",
            "odom"
        )

    # Create SLAM object
    def create_slam(self, Occu_grid_pub, slam_variables):
        window_size_pixel, size_m, OG_map_options, number_particles, tunning_options = slam_variables
        central_bar_width = 10
        turtlebot_L = 0.287
        self.my_slam = FastSlam(tunning_options, True, window_size_pixel, size_m, central_bar_width, OG_map_options, Occu_grid_pub, turtlebot_L, number_particles)
        self.count = 0

    # Calibrate the camera
    def calibrate_camera(self):
        dist = [0.1639958233797625, -0.271840030972792, 0.001055841660100477, -0.00166555973740089, 0.0]
        K = [322.0704122808738, 0.0, 199.2680620421962, 0.0, 320.8673986158544, 155.2533082600705, 0.0, 0.0, 1.0]
        P = [329.2483825683594, 0.0, 198.4101510452074, 0.0, 0.0, 329.1044006347656, 155.5057121208347, 0.0, 0.0, 0.0, 1.0, 0.0]
        
        # Adjust calibration parameters
        K[2] = 320
        K[5] = 240
        P[2] = 320
        P[6] = 240
        
        mtx = np.array(K).reshape(3, 3)
        dist = np.array(dist).reshape(1, 5)
        self.camera_matrix = mtx.astype(float)
        self.dist_coeffs = dist.astype(float)

    # Main loop to run the node
    def run(self):
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():  # Continue running until ROS node is shutdown
            self.publish_tf()  # Publish transformation frames
            self.my_slam.publish_landmarks()
            if self.rosbag_finished(start_time, self.k):  # Check if the rosbag playback has finished
                rospy.loginfo("Rosbag playback finished. Shutting down...")
                rospy.signal_shutdown("Rosbag playback finished")  # Shutdown ROS node
                break  # Exit the loop
            rospy.sleep(0.1)  # Process ROS callbacks once
        cv2.destroyAllWindows()  # Close OpenCV windows when node exits
        pygame.display.quit()
        pygame.quit()

    # Check if the rosbag playback has finished
    def rosbag_finished(self, start_time, duration):
        end_time = start_time + rospy.Duration.from_sec(duration)
        if rospy.Time.now() > end_time:
            return True  # Playback finished
        else:
            return False  # Playback ongoing

    # Get the trajectory from the SLAM object
    def get_trajectory(self):
        return self.my_slam.get_best_trajectory()
