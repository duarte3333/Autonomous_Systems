import numpy as np

from Simulation import SCREEN_WIDTH, SCREEN_HEIGHT
from micro_simulation.aux_1 import distance
from aux_slam import gauss_noise

class Particle:
    """ x: x position of the particle
        y: y position of the particle
        theta: orientation of the particle
        landmarks: list of landmarks in the map
        weight: weight of the particle"""
    def __init__(self, x, y,  orientation, is_turtlebot=False):
        self.pose = np.array([x, y, orientation])
        self.is_turtlebot = is_turtlebot
        self.landmarks = []
        self.weight = 1.0

    def apply_odometry(self, odometry):
        # This function updates the particle's pose based on odometry (motion model)
        noise_level = 0.1
        self.pose[0] += odometry[0] + np.random.normal(0, noise_level)
        self.pose[1] += odometry[1] + np.random.normal(0, noise_level)
        self.pose[2] += odometry[2] + np.random.normal(0, noise_level * 0.1)
        
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
