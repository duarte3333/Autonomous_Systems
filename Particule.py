import numpy as np

class Particle:
    def __init__(self, initial_pose):
        self.pose = np.array(initial_pose)
        self.landmarks = {}
        self.weight = 1.0

    def apply_odometry(self, odometry):
        # This function updates the particle's pose based on odometry (motion model)
        noise_level = 0.1
        self.pose[0] += odometry[0] + np.random.normal(0, noise_level)
        self.pose[1] += odometry[1] + np.random.normal(0, noise_level)
        self.pose[2] += odometry[2] + np.random.normal(0, noise_level * 0.1)
