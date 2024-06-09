import rospy
import cv2
import numpy as np
import cv2.aruco as aruco
import cv2 as CvBridgeError
from aux_slam import cart2pol

# Transform translation vector from camera coordinates to robot coordinates
def transform_camera_to_robot(translation_vector):
    # Rotation matrix from camera to robot frame (identity matrix if no rotation)
    R_cam_to_robot = np.array([
        [1, 0, 0],  
        [0, 1, 0],
        [0, 0, 1]
    ])
    # Translation vector from camera to robot frame
    T_cam_to_robot = np.array([0.076, 0, 0.103])  

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

# Compute the bearing angle from the translation vector
def compute_bearing_angle(tvec):
    x = tvec[0]
    z = tvec[2]
    bearing_angle_rad = np.arctan2(x, z)
    bearing_angle_deg = np.degrees(bearing_angle_rad)
    return bearing_angle_deg

# Callback function for image data
def image_callback(self, data):
    with self.lock:  # Ensure thread-safe operation with a lock
        self.current_aruco = []

        try:
            # Convert compressed image message to OpenCV format
            cv_image = self.bridge.compressed_imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))
            return

        # Convert the image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # Detect ArUco markers in the image
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        if ids is not None and len(ids) > 0:
            for i in range(len(ids)):
                marker_corners = corners[i][0]
                # Draw bounding box around detected markers
                cv2.polylines(cv_image, [np.int32(marker_corners)], True, (0, 255, 0), 2)
                # Estimate pose of each marker
                _, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.25, self.camera_matrix, self.dist_coeffs)

                # Draw a circle at the center of the image (for reference)
                cv2.circle(cv_image, (320, 240), radius=10, color=(255, 0, 0), thickness=-1)

                # Transform the translation vector to robot coordinates
                tvec = transform_camera_to_robot(tvec[0][0])
                # Convert Cartesian coordinates to polar coordinates
                dist, phi = cart2pol(tvec[0], tvec[2])

                # Fill the dictionary if the marker is detected
                if ids[i][0] not in self.dict:
                    self.dict[ids[i][0]] = []
                self.dict[ids[i][0]].append(phi)
                
                # Display the distance to the marker
                cv2.putText(cv_image, 'dist= ' + str(round(dist, 3)), 
                            (int(marker_corners[2][0] - 80), int(marker_corners[2][1]) + 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # Compute median of the last few measurements to reduce noise
                if len(self.dict[ids[i][0]]) >= 3:                
                    phi5 = -np.median(np.sort(self.dict[ids[i][0]][-4:-1]))  # Compute median
                    self.dict[ids[i][0]].pop(0)  # Remove the oldest measurement
                    cv2.putText(cv_image, 'ang=' + str(round(phi5, 3)), 
                                (int(marker_corners[1][0] - 70), int(marker_corners[1][1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    self.current_aruco.append((dist, phi5, ids[i][0]))
                else:
                    self.current_aruco.append((dist, -phi, ids[i][0]))

        # Compute SLAM with the detected ArUco markers
        self.my_slam.compute_slam(self.current_aruco)
        # Show the image with detected markers
        cv2.imshow('Aruco Detection', cv_image)
        cv2.waitKey(3)
