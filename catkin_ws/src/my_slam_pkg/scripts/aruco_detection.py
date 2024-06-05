#!/usr/bin/env python3
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
from metrics import compute_metrics
from odom_callback import odom_callback
from img_callback import image_callback
from aux_slam import get_rosbag_duration, cart2pol
from visualization_msgs.msg import Marker, MarkerArray


class ArucoSLAM:
    def odom_callback_wrapper(self, data):
        odom_callback(self, data)

    def image_callback_wrapper(self, data):
        image_callback(self, data)
        
    def __init__(self, rosbag_time):
        self.image_callback = image_callback
        self.odom_callback = odom_callback
        self.k = rosbag_time + 5
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.calibrate_camera()
        self.lock = threading.Lock()
        self.create_slam()
        rospy.loginfo('ArucoSLAM Node Started')
        rospy.init_node('aruco_slam') # Initialize the ROS node
        #subscribe node
        self.image_sub = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.image_callback_wrapper)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback_wrapper)
        self.landmark_pub = rospy.Publisher('/landmarks', MarkerArray, queue_size=10)
        self.current_aruco = []  
        self.odom =[0,0,0]
        self.bridge = CvBridge() # Initialize the CvBridge object

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
            quaternion = tf.transformations.quaternion_from_euler(0, 0, theta)
            br.sendTransform(
                (x, y, 0),
                quaternion,
                rospy.Time.now(),
                "base_link",
                "odom"
            )
    def create_slam(self):
        window_size_pixel=900    #tamanho da janela 
        sample_rate=5  #sample rate (Hz)
        size_m = 5#float(input('What should be the size of the map? n x n (in meters). n is: '))
        central_bar_width=10
        self.my_slam = FastSlam(True, window_size_pixel, sample_rate, size_m, central_bar_width, 0.287)
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
            self.publish_tf()
            self.my_slam.publish_landmarks()
            if self.rosbag_finished(start_time, self.k):  # Implement this function to check if the rosbag playback has finished
                rospy.loginfo("Rosbag playback finished. Shutting down...")
                rospy.signal_shutdown("Rosbag playback finished")  # Shutdown ROS node
                break  # Exit the loop
            rospy.sleep(0.1) # Process ROS callbacks once
        cv2.destroyAllWindows()  # Close OpenCV windows when node exits
    
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

def run_slam(rosbag_file):
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
                rviz = subprocess.Popen(['roslaunch', 'turtlebot3_gazebo', 'turtlebot3_gazebo_rviz.launch'])
                
                if not os.path.isfile(rosbag_file):
                    print(f"ERROR: The file {rosbag_file} does not exist.")
                    exit(1)
                
            rosbag_time = get_rosbag_duration(rosbag_file)
            slam = ArucoSLAM(rosbag_time)
            gt = 0 
            slam.run()
            # ate, rpe, mse_landmarks = compute_metrics(slam, gt, rosbag_nr)
            # print(f"Metrics for {rosbag_file}:")
            # print(f"ATE: {ate}, RPE: {rpe}, MSE Landmarks: {mse_landmarks}")
        finally:
            if rosbag_process:
                rosbag_process.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Select a rosbag to run.")
    parser.add_argument('rosbag', help="The rosbag file to run.")
    args = parser.parse_args()
    if args.rosbag.endswith('.bag'):
        print("entrei", args.rosbag)
        rosbag_file = f"../rosbag/{args.rosbag}"
    elif args.rosbag == 'live':
        rosbag_file = 'live'
    elif args.rosbag == 'microsim':
        rosbag_file = 'microsim'
    else:
        print("Invalid choice. Exiting.")
        exit(1)
    run_slam(rosbag_file)

    
    
    
""" 
if __name__ == '__main__':
    rosbag_time = int(input("Enter rosbag time \n"))
    rosbag_nr = int(input('Which Map is this? (1,2)'))
    slam = ArucoSLAM(rosbag_time)
    gt = 0 
    try:
        slam.run()
        gt += 0 #To implement ground_truth
    except KeyboardInterrupt:
        try:
            ate, rpe, mse_landmarks= compute_metrics(slam, gt, rosbag_nr)
            try:
                sys.exit(130)
            except SystemExit:
                os._exit(130)
        except KeyboardInterrupt:
            print('Exiting with no more metrics')
    
    
     """

"""


import rospy
import threading
from sensor_msgs.msg import CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from aruco_msgs.msg import MarkerArray
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import cv2
from cv2 import aruco
import copy

class ArucoSLAM:
    def __init__(self):
        self.calibrate_camera()
        self.lock = threading.Lock()
        self.create_slam()
        rospy.loginfo('ArucoSLAM Node Started')
        rospy.init_node('aruco_slam') # Initialize the ROS node

        # Subscribe to image and odometry topics
        self.image_sub = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.image_callback)
        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)

        # Subscribe to the marker array topic from aruco_ros
        self.aruco_marker_sub = rospy.Subscriber("/aruco_marker_publisher/markers", MarkerArray, self.aruco_markers_callback)

        self.current_aruco = []  
        self.odom = [0, 0, 0]
        self.bridge = CvBridge() # Initialize the CvBridge object

        self.dict = {15: [], 53: [], 60: [], 77: [], 100: []}
        self.map = {}

    def create_slam(self):
        window_size_pixel = 900    # Window size
        sample_rate = 5  # Sample rate (Hz)
        size_m = 5  # Size of the map in meters
        central_bar_width = 10
        self.my_slam = FastSlam(True, window_size_pixel, sample_rate, size_m, central_bar_width, 0.287)
        self.count = 0

    def calibrate_camera(self):
        dist = [0.1639958233797625, -0.271840030972792, 0.001055841660100477, -0.00166555973740089, 0.0]
        K = [322.0704122808738, 0.0, 199.2680620421962, 0.0, 320.8673986158544, 155.2533082600705, 0.0, 0.0, 1.0]
        mtx = np.array(K).reshape(3, 3)
        dist = np.array(dist).reshape(1, 5)

        self.camera_matrix = mtx.astype(float)
        self.dist_coeffs = dist.astype(float)
        
    def odom_callback(self, odom_data):
        with self.lock:
            x = odom_data.pose.pose.position.x
            y = odom_data.pose.pose.position.y
            xq = odom_data.pose.pose.orientation.x
            yq = odom_data.pose.pose.orientation.y
            zq = odom_data.pose.pose.orientation.z
            wq = odom_data.pose.pose.orientation.w
            quater = [xq, yq, zq, wq]
            self.odom = [x, y, quater]
            if self.count == 0:
                self.tara = copy.deepcopy(self.odom)
                self.count += 1

            self.odom[0] -= self.tara[0]
            self.odom[1] -= self.tara[1]
            self.my_slam.update_odometry(self.odom)
    
    def aruco_markers_callback(self, markers_data):
        with self.lock:
            self.current_aruco = []
            for marker in markers_data.markers:
                dist = np.sqrt(marker.pose.pose.position.x**2 + marker.pose.pose.position.y**2 + marker.pose.pose.position.z**2)
                angle_rad = np.arctan2(marker.pose.pose.position.x, marker.pose.pose.position.z)
                angle_deg = np.degrees(angle_rad)

                self.dict[marker.id].append(angle_deg)

                if len(self.dict[marker.id]) >= 5:                
                    phi5 = -np.median(np.sort(self.dict[marker.id][-6:-1]))
                    self.dict[marker.id].pop(0)
                    self.current_aruco.append((dist, phi5, marker.id))
                else:
                    self.current_aruco.append((dist, angle_deg, marker.id))
            self.my_slam.compute_slam(self.current_aruco)

    def image_callback(self, data):
        with self.lock:
            try:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {0}".format(e))
                return

            cv2.imshow('Aruco Detection', cv_image)
            cv2.waitKey(3)

    def run(self):
        rospy.spin()  # Keep the node running
        cv2.destroyAllWindows()

if __name__ == '__main__':
    slam = ArucoSLAM()
    slam.run()
"""
