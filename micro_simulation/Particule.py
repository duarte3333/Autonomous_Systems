import numpy as np
import math

from Simulation import SCREEN_WIDTH, SCREEN_HEIGHT
from aux_slam import gauss_noise
from aux_slam import multi_normal
from Landmark import Landmark
from numpy import linalg #this is for the inverse of a matrix

class Particle:
    """ x: x position of the particle
        y: y position of the particle
        theta: orientation of the particle
        landmarks: list of landmarks in the map
        weight: weight of the particle"""
        
    def __init__(self, pose,turtlebot_L,std_dev_motion=0.2 ,is_turtlebot=False):
        self.pose = pose
        self.is_turtlebot = is_turtlebot
        self.landmarks = {}
        self.weight = 1.0
        self.old_odometry = (0,0)
        self.std_dev_motion = std_dev_motion
        self.turtlebot_L=turtlebot_L
        self.observation_vector = np.zeros((2,1))
        
        self.J_matrix = np.zeros((2,2))
        self.adjusted_covariance = np.zeros((2,2))
        
    ##MOTION MODEL##
    def motion_model(self, odometry):
        """ This function updates the particle's pose based on odometry (motion model) """
        x, y, theta = self.pose
        deltaRight=odometry[0]-self.old_odometry[0]
        deltaLeft=odometry[1]-self.old_odometry[1]
        deltaD =(deltaRight + deltaLeft)/2
        delta_theta=(deltaRight - deltaLeft)/self.turtlebot_L
        delta_x=deltaD*math.cos(theta)
        delta_y=deltaD*math.sin(theta)
        noise=np.random.normal(0, self.std_dev_motion, 3)
        new_x = x + delta_x*(1+noise[1])
        new_y = y + delta_y*(1+noise[1])
        new_theta = (theta + delta_theta*(1+noise[2])) % (2 * np.pi)
        self.pose=np.array([new_x, new_y, new_theta])

        self.old_odometry=odometry

    ##WEIGHT##
    
    #[(distancia1, angle1, id), (distancia, angle, id)]
    def compute_weight(self, collected_data):
        """ data_values: list of tuples (distance, angle)"""
                        
        for landmark in collected_data:
            landmark_position_x, landmark_position_y, landmark_id = landmark
            if not str(landmark_id) in self.landmarks:
                self.create_landmark(landmark)
            #update Extended Kalman Filter
            self.compute_matrixs(self.landmarks[str(landmark_id)])
            self.update_landmark(landmark_position_x, landmark_position_y, landmark_id)
            
        #self.weight *= prob    
        
    # def associate_observation(self, observation):
    #     prob = 0
    #     landmark_index = -1
    #     for landmark_id, landmark in self.landmarks.items():
    #         predicted_obs = self.compute_matrixs(landmark)
    #         p = multi_normal(np.transpose(np.array([observation])), predicted_obs, self.adjusted_covariance)
    #         if (p > prob):
    #             prob = p
    #             landmark_index = landmark_id
                
    #     return prob, landmark_index
                
    def compute_matrixs(self, landmark):
        dx = landmark.x - self.pose.x
        dy = landmark.y - self.pose.y
        d2 = dx**2 + dy**2
        d = math.sqrt(d2)

        #predicted_obs = np.array([[d],[math.atan2(dy, dx)]])
        self.jacobian = np.array([[dx/d,   dy/d],
                             [-dy/d2, dx/d2]])
        self.adjusted_covariance = self.jacobian.dot(landmark.sigma).dot(np.transpose(self.jacobian)) + np.normal(0, 0.1, 2)
        #return predicted_obs
    
    ##UPDATE##
    def update_particle(self, odometry, collected_data):
        self.motion_model(odometry)
        self.compute_weight(collected_data)
        
    
    ##POSE##
    def get_pose(self):
        return (self.pose)
    
    # def set_pose(self, x, y, theta):
    #     if x > SCREEN_WIDTH:
    #         x = SCREEN_WIDTH
    #     if y > SCREEN_HEIGHT:
    #         y = SCREEN_HEIGHT
    #     self.pose = np.array([x, y, theta])
    
    ##LANDMARKS##
    
    def get_landmark(self, landmark_id):
        return self.landmarks[str(landmark_id)]
    
    def create_landmark(self, observation):
        distance, angle, id = observation
        x = self.pose[0] + distance * math.cos(angle + self.pose[2])
        y = self.pose[1] + distance * math.sin(angle + self.pose[2])
        self.landmarks[id] = Landmark(x, y)
        
    def update_landmark(self, obs, landmark_idx, ass_obs, ass_jacobian, ass_adjcov):
        landmark = self.landmarks[landmark_idx]
        K_matrix = landmark.sig.dot(np.transpose(self.J_matrix)).dot(linalg.inv(ass_adjcov))
        new_mu = landmark.mu + K_matrix.dot(obs - ass_obs)
        new_sig = (np.eye(2) - K_matrix.dot(self.J_matrix)).dot(landmark.sigma)
        landmark.update(new_mu, new_sig)
        
    
    def update_landmark(self, observation, landmark_id):
        ###FALTA FAZERRR###