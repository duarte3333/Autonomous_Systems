import math
import numpy as np

class Landmark:
    def __init__(self, x, y):
        self.x = x  # X-coordinate of the landmark
        self.y = y  # Y-coordinate of the landmark
        self.id = id  # Unique identifier for the landmark
        
        # Mean (or expected value) in the probability distribution describing the landmark's position
        self.mu = np.array([[self.x], [self.y]])  # Column vector for the mean
        
        # Covariance matrix associated with the landmark (3x3)
        # Smaller values imply higher confidence in the position estimate
        self.sigma = np.eye(3) * 1000  # Initialize with high uncertainty

    # Calculate the Euclidean distance to another point (x, y)
    def distance_to(self, x, y):
        return math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)

    # Update the mean and covariance of the landmark
    def update(self, mu, sigma):
        self.mu = mu  # Update mean
        self.sigma = sigma  # Update covariance
        # Optionally update the x and y coordinates to reflect the mean
        # self.x = self.mu[0][0]
        # self.y = self.mu[1][0]

    # String representation of the landmark for easy debugging
    def __str__(self):
        return f"Landmark(x={self.x}, y={self.y})"
