import numpy as np
import math

from Landmark import Landmark
from numpy import linalg #this is for the inverse of a matrix

class Particle:
    """ x: x position of the particle
        y: y position of the particle
        theta: orientation of the particle
        landmarks: list of landmarks in the map
        weight: weight of the particle"""
        
    def __init__(self, pose,nr_particles,turtlebot_L,std_dev_motion=0.2 ,is_turtlebot=False):
        self.pose = pose
        self.is_turtlebot = is_turtlebot
        self.landmarks = {}
        self.weight = 1.0
        self.std_dev_motion = std_dev_motion
        self.turtlebot_L=turtlebot_L
        self.observation_vector = np.zeros((2,1))
        self.default_weight=1/nr_particles
        self.J_matrix = np.zeros((2,2))
        self.adjusted_covariance = np.zeros((2,2))
        
    ##MOTION MODEL##
    def motion_model(self, odometry_delta):
        """ This function updates the particle's pose based on odometry (motion model) """
        x, y, theta = self.get_pose()
        #odometry = np.random.normal(0, self.std_dev_motion, 2)
        #print(self.old_odometry, ' after calling')

        #print('Odometry: ',odometry[0],', ',  odometry[1] ) 
        #print('Odometry: ',self.old_odometry[0],', ',  self.old_odometry[1] , 'old') 

        #deltaRight=odometry[0]-self.old_odometry[0]
        #deltaLeft=odometry[1]-self.old_odometry[1]
        deltaLeft = odometry_delta[0]
        deltaRight = odometry_delta[1]
        #print('deltas: ',deltaRight,', ',  deltaLeft ) 
      
        deltaD =(deltaRight + deltaLeft)/2
        delta_theta=-(deltaRight - deltaLeft)/self.turtlebot_L#aqui tinha um menos
        delta_x=deltaD*math.cos(theta)
        delta_y=-deltaD*math.sin(theta)
        noise=np.random.normal(0, self.std_dev_motion, 3)
        new_x = x + delta_x*(1+noise[0])
        new_y = y + delta_y*(1+noise[1])
        new_theta = (theta + delta_theta*(1+noise[2])) % (2 * np.pi)
        #print("Particle pose delta x, y",delta_x*(1+noise[0]),delta_y*(1+noise[1]) )
        #print('Odometry: ',deltaRight,', ',  deltaLeft ) 
        self.pose=np.array([new_x, new_y, new_theta])

        #self.old_odometry=odometry.copy()

    # ##WEIGHT##
    # #[(distancia1, angle1, id), ... , (distanciaN, angleN, idN)]
    # def compute_weight(self, collected_data):
    #     """ collected_data: list of tuples (distance, angle, id)"""
    #     self.weight = 1.0 #reset weight 
    #     for distance, angle_diff, landmark_id in collected_data:
    #         landmark_id = str(landmark_id)
    #         if landmark_id not in self.landmarks:
    #             self.create_landmark(distance, angle_diff, landmark_id)
    #         else:
    #             #update Extended Kalman Filter
    #             self.update_landmark(distance, angle_diff, landmark_id)



 ##WEIGHT##
    def handle_landmark(self, landmark_dist, landmark_bearing_angle, landmark_id):
        landmark_id = str(landmark_id)
        if landmark_id not in self.landmarks:
            self.create_landmark(landmark_dist, landmark_bearing_angle, landmark_id)
        else:
            #update Extended Kalman Filter
            self.update_landmark(landmark_dist, landmark_bearing_angle, landmark_id)


    def create_landmark(self, distance, angle, landmark_id):
        x, y, theta = self.get_pose()
        landmark_x = x + distance * math.cos(theta + angle)
        landmark_y = y - distance * math.sin(theta + angle)
        self.landmarks[landmark_id] = Landmark(landmark_x, landmark_y)
        
        # Prediction of the measurement
        dx = landmark_x - x
        dy = landmark_y - y
        predicted_distance = math.sqrt(dx**2 + dy**2)
        predicted_angle = math.atan2(dy, dx) -theta
        """  try:
            predicted_angle=predicted_angle[0]# to make it not be an array, but a value
        except:
            predicted_angle=predicted_angle """
        # Calculate Jacobian matrix H of the measurement function
        q = dx**2 + dy**2
        sqrt_q = math.sqrt(q)
        # dx=dx[0]
        # dy=dy[0]
        # q=q[0]

        J = np.array([[dx / sqrt_q, dy / sqrt_q],[-dy / q, dx / q]])
        J = J.reshape(2, 2)
        
        # Measurement noise covariance matrix (should be tuned)
        Q = np.diag([0.1, 0.1])  # Example values , Q is Q_t in the book
        
        self.landmarks[landmark_id].sigma = linalg.inv(J) @ Q @ linalg.inv(J).T
        
        #set a default importance weight
        self.weight = self.default_weight#p0 in the book
        
    
    def update_landmark(self, distance, angle, landmark_id):
            """Updates an existing landmark using the EKF update step."""
            landmark = self.landmarks[str(landmark_id)]
            x, y, theta = self.pose
            landmark_x = landmark.x
            landmark_y = landmark.y
            #landmark_x = x + distance * math.cos(theta + angle)
            #landmark_y = y + distance * math.sin(theta + angle)
            
            # Prediction of the measurement
            dx = landmark_x - x
            dy = landmark_y - y
            predicted_distance = math.sqrt(dx**2 + dy**2)
            predicted_angle = -math.atan2(dy, dx) -theta
            # Calculate Jacobian matrix H of the measurement function
            q = dx**2 + dy**2
            sqrt_q = math.sqrt(q)
            
            J = np.array([[dx / sqrt_q, dy / sqrt_q],[dy / q, -dx / q]])
            J = J.reshape(2, 2)
            
            # Measurement noise covariance matrix (should be tuned)
            Q = np.diag([0.1, 0.1])  # Example values

            # Calculate the Kalman Gain
            S = J @ landmark.sigma @ J.T + Q  # Measurement prediction covariance
            K = landmark.sigma @ J.T @ np.linalg.inv(S) #S is Q in the book
            # Innovation (measurement residual)
            innovation = np.array([distance - predicted_distance, angle - predicted_angle])
    
            # Update landmark state
            update = K @ innovation
            landmark.x += update[0]
            landmark.y += update[1]

            # Update the covariance
            I = np.eye(2)  # Identity matrix
            landmark.sigma = (I - K @ J) @ landmark.sigma

            # Update the weight using the measurement likelihood
            det_S = np.linalg.det(S)
            if det_S > 0:
                weight_factor = 1 / np.sqrt(2 * np.pi * det_S)
                exponent = -0.5 * innovation.T @ np.linalg.inv(S) @ innovation
                self.weight *= weight_factor * np.exp(exponent)
        
    
    ##UPDATE##
    def update_particle(self, collected_data):
        self.compute_weight(collected_data)
        
    
    ##POSE##
    def get_pose(self):
        return (self.pose)
    
    
