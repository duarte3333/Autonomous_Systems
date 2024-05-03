#!/usr/bin/env python
import rospy
import cv2
import cv2.aruco as aruco
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge #package that converts between ROS Image messages and OpenCV Image formats
import yaml
import os

class ArucoSLAM:
    def __init__(self):
        self.calibrate_camera()
        rospy.init_node('aruco_slam') # Initialize the ROS node
        self.image_sub = rospy.SubscribeListener("/camera/image_raw", Image, self.image_callback)
        # Calibrate the cameraiber("/camera/image_raw", Image, self.image_callback)
        self.bridge = CvBridge() # Initialize the CvBridge object

        #the tag of our aruco dictionary is 4X4_100
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_100)
        
        self.parameters = aruco.DetectorParameters_create()
        self.map = {} 
    
    def calibrate_camera(self):
        #chessboard_size = (6, 9)  # Adjust to your chessboard, (columns, rows)
        #square_size = 0.0245  # Chessboard square size in meters
        
        # Prepare object points
        #objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        #objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
        
        # Arrays to store object points and image points
        #objpoints = []  # 3d points in real world space
        #imgpoints = []  # 2d points in image plane
        
        # Collect samples
        #print("Collecting images for calibration. Press 'q' to stop collecting and start calibration.")
        #while True:
          #  ret, frame = self.capture.read()
           # if ret:
            #    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
             #   ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
              #  if ret:
               #     objpoints.append(objp)
                #    imgpoints.append(corners)
                 #   #  Draw and display the corners
                  #  cv2.drawChessboardCorners(frame, chessboard_size, corners, ret)
                   # cv2.imshow('Frame', frame)
                    #if cv2.waitKey(500) & 0xFF == ord('q'):
                     #   break
        #cv2.destroyAllWindows()
        
        # Calibration
        #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
       # data = {'camera_matrix': mtx.tolist(), 'dist_coeff': dist.tolist()}
        #with open('calibration.yaml', 'w') as f:
        #    yaml.dump(data, f)


        dist = np.matrix('[0.1639958233797625, -0.271840030972792, 0.001055841660100477, -0.00166555973740089, 0.0]')
        mtx = np.matrix('[[329.2483825683594, 0.0, 198.4101510452074, 0.0],[0.0, 329.1044006347656, 155.5057121208347, 0.0],[0.0, 0.0, 1.0, 0.0]]')

        self.camera_matrix = mtx.astype(float)
        self.dist_coeffs = dist.astype(float)
        
    def image_callback(self, data): # This function is called whenever a new image is published to the camera topic
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        
        if ids is not None:
            # Here you would calculate the robot's pose based on the detection of markers
            rospy.loginfo('Markers Detected: %s', ids)
            
            marker_length = 0.1  # Marker side length in meters
            rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_length, self.camera_matrix, self.dist_coeffs)
            
            for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                rospy.loginfo("Rotation Vector:\n%s\nTranslation Vector:\n%s", rvec, tvec)
                
                # Store or update the marker's position in the map
                if ids[i][0] not in self.map:
                    self.map[ids[i][0]] = tuple(tvec[0])
                    
            rospy.loginfo("Current Map: %s", self.map)

    def run(self):
        rospy.spin() # Keep the node running

if __name__ == '__main__':
    slam = ArucoSLAM()
    slam.run()
