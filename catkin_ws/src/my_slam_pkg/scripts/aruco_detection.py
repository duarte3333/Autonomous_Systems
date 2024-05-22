#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from sensor_msgs.msg import CompressedImage
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge #package that converts between ROS Image messages and OpenCV Image formats
import yaml
import os
from cv_bridge import CvBridge, CvBridgeError
from FastSlam import FastSlam
import threading
import copy
import math


def compute_marker_size_in_pixels(marker_corners):
    _, _, w, h = cv2.boundingRect(marker_corners)
    marker_size_pixels = max(w, h)
    return marker_size_pixels
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)
        
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
        size_m = 3#float(input('What should be the size of the map? n x n (in meters). n is: '))
        central_bar_width=10
        self.my_slam = FastSlam(True, window_size_pixel, sample_rate, size_m, central_bar_width, 0.287)
        self.count = 0

    def calibrate_camera(self):

        dist =[0.1639958233797625, -0.271840030972792, 0.001055841660100477, -0.00166555973740089, 0.0]
        K = [322.0704122808738, 0.0, 199.2680620421962, 0.0, 320.8673986158544, 155.2533082600705, 0.0, 0.0, 1.0]
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
                    _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.1, self.camera_matrix, self.dist_coeffs)
                    homography, _ = cv2.findHomography(world_coords, marker_corners, cv2.RANSAC,7)
                    essential_matrix = np.dot(np.dot(np.linalg.inv(self.camera_matrix), homography),self.camera_matrix)
                    u, _, _ = np.linalg.svd(essential_matrix)

                    dpixels = compute_marker_size_in_pixels(marker_corners)
                    dist = calculate_distance(dpixels, 322.0704122808738) #515.2
                    #f = 0.28*dpixels*10

                    translation_vector = u[:, -1]
                    translation_vector /= np.linalg.norm(translation_vector)
                    _,phi = cart2pol(translation_vector[2],translation_vector[0])
                    phi = phi *180 /np.pi

                    self.dict[ids[i][0]].append(phi)

                    # ID of the marker
                    #cv2.putText(cv_image, str(ids[i][0]), (int(marker_corners[0][0] - 10), int(marker_corners[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    cv2.putText(cv_image, 'dist= ' + str(round(dist, 3)), (int(marker_corners[2][0] - 80), int(marker_corners[2][1]) + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    if len(self.dict[ids[i][0]]) >= 5:                
                        phi5 =  np.median(np.sort(self.dict[ids[i][0]][-6:-1]))
                        self.dict[ids[i][0]].pop(0)
                        cv2.putText(cv_image,  'ang='+str(round(phi5,3)), (int(marker_corners[1][0]-70),int(marker_corners[1][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255 ), 3)
                        self.current_aruco.append((dist,phi5,ids[i][0]))
                    else:
                        self.current_aruco.append((dist,phi,ids[i][0]))
            self.my_slam.compute_slam(self.current_aruco)
                #rospy.loginfo("IDs detected: %s", ids)  # Correct logging of detected IDs
            cv2.imshow('Aruco Detection', cv_image)
            cv2.waitKey(3)

    def run(self):
        rospy.spin() # Keep the node running
        cv2.destroyAllWindows() 

if __name__ == '__main__':
    slam = ArucoSLAM()
    slam.run()
