import math
import numpy as np

class Landmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.id = id
        
        #acts as the mean (or expected value) in the probability distribution describing the landmark's position
        self.mu = np.array([[self.x],[self.y]]) #column vector
        
        #smaller value in the covariance matrix implies higher confidence in the position estimate
        self.sigma = np.eye(3) *1000 #2x2 covariance matrix associated with the landmark
        
        
    def distance_to(self, x, y):
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    def update(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.x = self.mu[0][0]
        self.y = self.mu[1][0]
        
    def __str__(self):
        return f"Landmark(x={self.x}, y={self.y})"