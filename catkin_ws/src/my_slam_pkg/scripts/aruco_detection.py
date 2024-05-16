#!/usr/bin/env python3
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge #package that converts between ROS Image messages and OpenCV Image formats
import yaml
import os
from cv_bridge import CvBridge, CvBridgeError

class ArucoSLAM:
    def __init__(self):
        self.calibrate_camera()
        rospy.loginfo('ArucoSLAM Node Started')
        rospy.init_node('aruco_slam') # Initialize the ROS node
       
        #subscribe node
        self.image_sub = rospy.Subscriber("/raspicam_node/image/compressed", CompressedImage, self.image_callback)
        # Calibrate the cameraiber("/camera/image_raw", Image, self.image_callback)
        self.bridge = CvBridge() # Initialize the CvBridge object

        #the tag of our aruco dictionary is 4X4_100
        
        #self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100) 
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)

        #Do not change the tag used (cv2.aruco.DICT_4X4_100), only if we change the ArUcos we're using (and thus their tags) 

        #self.parameters =  cv2.aruco.DetectorParameters()
        self.parameters = aruco.DetectorParameters_create()
        #self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.dict = {15: [], 53: [], 60: [], 77: [], 100: []}
        self.map = {} 
    
    def calibrate_camera(self):

        dist =[0.1639958233797625, -0.271840030972792, 0.001055841660100477, -0.00166555973740089, 0.0]
        K = [322.0704122808738, 0.0, 199.2680620421962, 0.0, 320.8673986158544, 155.2533082600705, 0.0, 0.0, 1.0]
        mtx = np.array(K).reshape(3, 3)
        dist = np.array(dist).reshape(1,5)

        self.camera_matrix = mtx.astype(float)
        self.dist_coeffs = dist.astype(float)
        
    def image_callback(self, data):
       
        def compute_marker_size_in_pixels(marker_corners):
            _, _, w, h = cv2.boundingRect(marker_corners)
            marker_size_pixels = max(w, h)
            return marker_size_pixels
        
        def cart2pol(x, y):
            rho = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)
            return(rho, phi)
        
        def calculate_distance(marker_size_pixels, focal_length):
            distance = (0.1 * focal_length) / marker_size_pixels
            return distance

        marker_length = 0.1  #length of the marker in meters, change this if we use another marker 
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
                dist = calculate_distance(dpixels, 515.2)
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
                    phi5 =  np.median(self.dict[ids[i][0]][-6:-1])
                    self.dict[ids[i][0]].pop(0)
                    cv2.putText(cv_image,  'ang='+str(round(phi5,3)), (int(marker_corners[1][0]-70),int(marker_corners[1][1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255 ), 3)

            rospy.loginfo("IDs detected: %s", ids)  # Correct logging of detected IDs
        cv2.imshow('Aruco Detection', cv_image)
        cv2.waitKey(3)

    def run(self):
        rospy.spin() # Keep the node running
        cv2.destroyAllWindows() 
if __name__ == '__main__':
    slam = ArucoSLAM()
    slam.run()
