import numpy as np
import math

from Simulation import SCREEN_WIDTH, SCREEN_HEIGHT
from micro_simulation.aux_1 import distance
from aux_slam import gauss_noise

class Particle:
    """ x: x position of the particle
        y: y position of the particle
        theta: orientation of the particle
        landmarks: list of landmarks in the map
        weight: weight of the particle"""
    def __init__(self, pose,turtlebot_L,std_dev_motion=0.2 ,is_turtlebot=False):
        self.pose = pose
        self.is_turtlebot = is_turtlebot
        self.landmarks = []
        self.weight = 1.0
        self.old_odometry = (0,0)
        self.std_dev_motion = std_dev_motion
        self.turtlebot_L=turtlebot_L

    def apply_odometry(self, odometry):
        """ # This function updates the particle's pose based on odometry (motion model)
        noise_level = 0.1
        self.pose[0] += odometry[0] + np.random.normal(0, noise_level)
        self.pose[1] += odometry[1] + np.random.normal(0, noise_level)
        self.pose[2] += odometry[2] + np.random.normal(0, noise_level * 0.1)
         """
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




    def get_pose(self):
        return (self.pose.x, self.pose.y)
    
    def set_pose(self, x, y, theta):
        if x > SCREEN_WIDTH:
            x = SCREEN_WIDTH
        if y > SCREEN_HEIGHT:
            y = SCREEN_HEIGHT
        self.pose = np.array([x, y, theta])
    
    def check_pos
    def forward(self, distance):
        self.pose[0] += distance * np.cos(self.pose[2]) + gauss_noise(0, 0.1)
        self.pose[1] += distance * np.sin(self.pose[2])
