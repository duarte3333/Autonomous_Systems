import rospy
import cv2
import numpy as np
import cv2.aruco as aruco
import cv2 as CvBridgeError
from aux_slam import cart2pol

def transform_camera_to_robot(translation_vector):
    R_cam_to_robot = np.array([
    [1, 0, 0],  
    [0, 1, 0],
    [0, 0, 1]
    ])
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

def compute_bearing_angle(tvec):
        x = tvec[0]
        z = tvec[2]
        bearing_angle_rad = np.arctan2(x, z)
        bearing_angle_deg = np.degrees(bearing_angle_rad)
    
        return bearing_angle_deg
def image_callback(self, data):
    with self.lock:
        self.current_aruco = []
        # marker_length = 0.25  #length of the marker in meters, change this if we use another marker 
        # world_coords = np.array([[-marker_length/2, -marker_length/2, 0],
        #                     [marker_length/2, -marker_length/2, 0],
        #                     [marker_length/2, marker_length/2, 0],
        #                     [-marker_length/2, marker_length/2, 0]], dtype=np.float32)
        
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
                dist,phi = cart2pol(tvec[0],tvec[2])

                # Fill the dictionary if the marker is detected
                if ids[i][0] not in self.dict:
                    self.dict[ids[i][0]] = []
                self.dict[ids[i][0]].append(phi)
                
                #cv2.putText(cv_image, str(ids[i][0]), (int(marker_corners[0][0] - 100), int(marker_corners[0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
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
