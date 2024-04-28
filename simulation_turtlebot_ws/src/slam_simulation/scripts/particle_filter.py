import numpy as np

#Each particle should have a pose and a map. 
class Particle:
    def __init__(self):
        self.pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.map = np.zeros((100, 100))  # Example: simple grid map
