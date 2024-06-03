#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge, CvBridgeError
from FastSlam import FastSlam
import threading
import copy
from metrics import compute_metrics

def transform_camera_to_robot(translation_vector):
    R_cam_to_robot = np.array([
    [1, 0, 0],  # Replace these with the actual rotation matrix values
    [0, 1, 0],
    [0, 0, 1]
    ])
    T_cam_to_robot = np.array([0.076, 0, 0.103])  # Replace this with the actual translation vector

    # Function to trans
    # Convert translation_vector to homogeneous coordinates
    translation_vector_hom = np.append(translation_vector, [1])
    
    # Create a transformation matrix from R and T
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R_cam_to_robot
    transformation_matrix[:3, 3] = T_cam_to_robot
    
    # Apply the transformation
    translation_vector_robot_hom = np.dot(transformation_matrix, translation_vector_hom)
    
    # Convert back to 3D coordinates
    translation_vector_robot = translation_vector_robot_hom[:3]
    
    return translation_vector_robot

def compute_marker_size_in_pixels(marker_corners):
    _, _, w, h = cv2.boundingRect(marker_corners)
    marker_size_pixels = max(w, h)
    return marker_size_pixels
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
    
def compute_bearing_angle(tvec):
        x = tvec[0]
        z = tvec[2]
        bearing_angle_rad = np.arctan2(x, z)
        bearing_angle_deg = np.degrees(bearing_angle_rad)
    
        return bearing_angle_deg

def calculate_distance(marker_size_pixels, focal_length):
    distance = (0.25 * focal_length) / marker_size_pixels
    return distance

class ArucoSLAM:
    def __init__(self):
        self.calibrate_camera()
        self.lock = threading.Lock()
        self.create_slam()
        rospy.loginfo('ArucoSLAM Node Started')
        rospy.init_node('aruco_slam') # Initialize the ROS node
        #subscribe node
        self.image_sub = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.image_callback)
        self.current_aruco = []  

        self.odom_sub = rospy.Subscriber("/odom", Odometry, self.odom_callback)
        self.odom =[0,0,0]
        # Calibrate the cameraiber("/camera/image_raw", Image, self.image_callback)
        self.bridge = CvBridge() # Initialize the CvBridge object

        #the tag of our aruco dictionary is 4X4_100
        try:
            self.aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
        except:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250) 

        #Do not change the tag used (cv2.aruco.DICT_4X4_100), only if we change the ArUcos we're using (and thus their tags) 
        try:
            self.parameters = aruco.DetectorParameters_create()
        except:
            self.parameters =  cv2.aruco.DetectorParameters()
        #self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.dict = {15: [], 53: [], 60: [], 77: [], 100: []}
        self.map = {} 


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

        K[2] = 320  # K[1,2]
        K[5] = 240  # K[2,2]

        P[2] = 320  # Tx
        P[6] = 240  # Ty
        
        mtx = np.array(K).reshape(3, 3)
        dist = np.array(dist).reshape(1,5)

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
            quater = [xq,yq,zq,wq]
            self.odom = [x,y,quater]
            if self.count == 0:
                self.tara = copy.deepcopy(self.odom)
                self.count +=1

            self.odom[0] -= self.tara[0]
            self.odom[1] -= self.tara[1]
            self.my_slam.update_odometry(self.odom)
    
    def image_callback(self, data):
        with self.lock:
            self.current_aruco = []
            marker_length = 0.25  #length of the marker in meters, change this if we use another marker 
            world_coords = np.array([[-marker_length/2, -marker_length/2, 0],
                                [marker_length/2, -marker_length/2, 0],
                                [marker_length/2, marker_length/2, 0],
                                [-marker_length/2, marker_length/2, 0]], dtype=np.float32)
            
            try:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {0}".format(e))
                return

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            if ids is not None and len(ids) > 0:
                for i in range(len(ids)):
                    marker_corners = corners[i][0]
                    cv2.polylines(cv_image, [np.int32(marker_corners)], True, (0, 255, 0), 2)  # Bounding Box
                    _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.25, self.camera_matrix, self.dist_coeffs)

                    #CIRCLES
                    #teste = np.matmul(self.camera_teste, np.array(np.append(tvec,1)).reshape(4,1))
                    #centroid = np.mean(marker_corners, axis=0)
                    #cv2.circle(cv_image, (int(teste[0][0]/teste[2][0]),int(teste[1][0]/teste[2][0])), radius=10, color=(0, 0, 255), thickness=-1)
                    cv2.circle(cv_image, (320,240), radius=10, color=(255, 0, 0), thickness=-1)

                    tvec = transform_camera_to_robot(tvec[0][0])
                    phi = compute_bearing_angle(tvec)
                    dist = tvec[2]
                    self.dict[ids[i][0]].append(phi)
                   # cv2.putText(cv_image, str(ids[i][0]), (int(marker_corners[0][0] - 100), int(marker_corners[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(cv_image, 'dist= ' + str(round(dist, 3)), (int(marker_corners[2][0] - 80), int(marker_corners[2][1]) + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    if len(self.dict[ids[i][0]]) >= 3:                
                        phi5 =  -np.median(np.sort(self.dict[ids[i][0]][-4:-1])) #I HAVE CHANGED THIS SIGN
                        self.dict[ids[i][0]].pop(0)
                        cv2.putText(cv_image,  'ang='+str(round(phi5,3)), (int(marker_corners[1][0]-70),int(marker_corners[1][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255 ), 3)
                        self.current_aruco.append((dist,phi5,ids[i][0]))
                    else:
                        self.current_aruco.append((dist,-phi,ids[i][0]))
            self.my_slam.compute_slam(self.current_aruco)
                #rospy.loginfo("IDs detected: %s", ids)  # Correct logging of detected IDs
            cv2.imshow('Aruco Detection', cv_image)
            cv2.waitKey(3)
    """
    def image_callback(self, data):
        with self.lock:
            self.current_aruco = []
            marker_length = 0.25  # Length of the marker in meters

            try:
                cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                rospy.logerr("CvBridge Error: {0}".format(e))
                return

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            if ids is not None and len(ids) > 0:
                for i in range(len(ids)):
                    marker_corners = corners[i][0]
                    cv2.polylines(cv_image, [np.int32(marker_corners)], True, (0, 255, 0), 2)  # Bounding Box
                    
                    # Estimate pose of each marker
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners[i], marker_length, self.camera_matrix, self.dist_coeffs)
                    
                    # Draw axis for each marker
                   # aruco.drawAxis(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[0], tvecs[0], marker_length * 0.5)
                    
                    # Compute the distance to the marker
                    dist = np.linalg.norm(tvecs[0][0])
                    
                    # Calculate the angle of the marker relative to the camera
                    tvec = tvecs[0][0]
                    angle_rad = np.arctan2(tvec[0], tvec[2])  # Angle in radians
                    angle_deg = np.degrees(angle_rad)  # Convert to degrees
                    
                    self.dict[ids[i][0]].append(angle_deg)

                    # ID of the marker
                    cv2.putText(cv_image, str(ids[i][0]), (int(marker_corners[0][0] - 10), int(marker_corners[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(cv_image, 'dist= ' + str(round(dist, 3)), (int(marker_corners[2][0] - 80), int(marker_corners[2][1]) + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    if len(self.dict[ids[i][0]]) >= 5:                
                        phi5 = -np.median(np.sort(self.dict[ids[i][0]][-6:-1]))
                        self.dict[ids[i][0]].pop(0)
                        cv2.putText(cv_image, 'ang=' + str(round(-phi5, 3)), (int(marker_corners[1][0] - 70), int(marker_corners[1][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        self.current_aruco.append((dist, phi5, ids[i][0]))
                    else:
                        self.current_aruco.append((dist, angle_deg, ids[i][0]))
            self.my_slam.compute_slam(self.current_aruco)
            cv2.imshow('Aruco Detection', cv_image)
            cv2.waitKey(3)
    """   
    
    
    # def run(self):
    #     rospy.spin() # Keep the node running
    #     cv2.destroyAllWindows() 
        
    def run(self):
        start_time = rospy.Time.now() 
        while not rospy.is_shutdown():  # Continue running until ROS node is shutdown
            # Check if the rosbag playback has finished
            if self.rosbag_finished(start_time,105):  # Implement this function to check if the rosbag playback has finished
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

if __name__ == '__main__':
    slam = ArucoSLAM()
    slam.run()
    ground = 0 #To implement
    ate, rpe = compute_metrics(slam, ground)

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
