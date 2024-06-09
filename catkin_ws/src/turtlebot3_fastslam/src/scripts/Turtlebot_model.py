import math
import numpy as np
import random
from aux_slam import *

class TurtleBot3Waffle:
    def __init__(self, time_interval, Odometry_noise, width_meters, height_meters, turtlebot_radius, x=0, y=0, theta=0):
        """
        Initialize the TurtleBot3 Waffle simulation parameters.
        
        Args:
            time_interval: Time interval for motion updates.
            Odometry_noise: Boolean indicating whether to add noise to odometry.
            width_meters: Width of the simulation area in meters.
            height_meters: Height of the simulation area in meters.
            turtlebot_radius: Radius of the TurtleBot.
            x: Initial x position.
            y: Initial y position.
            theta: Initial orientation (in radians).
        """
        self.delta_time = time_interval
        self.Odometry_noise = Odometry_noise
        self.x = x  # x position
        self.y = y  # y position
        self.theta = theta  # orientation (in radians)
        self.wheel_base = 0.287  # Distance between wheels
        self.wheel_radius = 0.033  # Radius of the wheels
        self.maxLinearVel = 0.26  # Maximum linear velocity
        self.maxAngularVel = 1.82  # Maximum angular velocity (rad/s)
        self.width_meters = width_meters 
        self.height_meters = height_meters
        self.turtlebot_radius = turtlebot_radius

        self.odometry_left = 0  # Left wheel odometry
        self.odometry_right = 0  # Right wheel odometry
        self.Camera_fieldView = 62.2  # Field of view of the camera (in degrees)
        self.Camera_maxDist = 2  # Maximum distance that camera can detect a marker with quality (in meters)

        # Parameters for noise in the odometry
        self.mean = 0  # Mean of the Gaussian distribution
        self.std_dev = 0.01  # Standard deviation of the Gaussian distribution

    # Method to move the TurtleBot with specified linear and angular velocities
    def move(self, linear_velocity, angular_velocity, delta_time):
        if not delta_time:
            delta_time = self.delta_time
            
        # Check if velocity is out of range
        if abs(angular_velocity) > self.maxAngularVel:
            angular_velocity = self.maxAngularVel * angular_velocity / abs(angular_velocity)  # Maintains sign of angular velocity
        if abs(linear_velocity) > self.maxLinearVel:
            linear_velocity = self.maxLinearVel * linear_velocity / abs(linear_velocity)  # Maintains sign of linear velocity

        old_x = self.x
        old_y = self.y
        
        # Kinematics calculations
        v_x = linear_velocity * math.cos(self.theta)
        v_y = linear_velocity * math.sin(self.theta)
        self.x += v_x * delta_time
        self.y -= v_y * delta_time 
        self.theta += angular_velocity * delta_time
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))  # Normalize theta between -pi and pi
        
        # Check that TurtleBot is inside the screen boundaries
        if self.x + self.width_meters / 2 > self.width_meters - self.turtlebot_radius:
            self.x = self.width_meters / 2 - self.turtlebot_radius
        elif self.x + self.width_meters / 2 < 0 + self.turtlebot_radius:
            self.x = -self.width_meters / 2 + self.turtlebot_radius

        if self.y + self.height_meters / 2 > self.height_meters - self.turtlebot_radius:
            self.y = self.height_meters / 2 - self.turtlebot_radius
        elif self.y + self.height_meters / 2 < 0 + self.turtlebot_radius:
            self.y = -self.height_meters / 2 + self.turtlebot_radius
            
        # Odometry calculations
        Translation = math.copysign(math.sqrt((self.x - old_x)**2 + (self.y - old_y)**2), linear_velocity)
        Rotation = angular_velocity * delta_time
        odometry_left = Translation - Rotation * self.wheel_base / 2  # This is given in meters
        odometry_right = Translation + Rotation * self.wheel_base / 2  # This is given in meters
        gaussian_noise_left = np.random.normal(0, self.std_dev, 1) * odometry_left
        gaussian_noise_left = gaussian_noise_left[0]
        gaussian_noise_right = np.random.normal(0, self.std_dev, 1) * odometry_right
        gaussian_noise_right = gaussian_noise_right[0]

        if not self.Odometry_noise:
            gaussian_noise_left = 0
            gaussian_noise_right = 0
        
        self.odometry_left += odometry_left + gaussian_noise_left
        self.odometry_right += odometry_right + gaussian_noise_right

    # Method to check which landmarks are within the TurtleBot's field of view
    def check_landmarks(self, landmarks):
        indices_in_sight = []
        my_x = self.x + self.width_meters / 2
        my_y = self.y + self.height_meters / 2
        for i, landmark in enumerate(landmarks):
            landmark_x, landmark_y = landmark
            # Calculate distance between camera and landmark
            distance = math.sqrt((my_x - landmark_x)**2 + (my_y - landmark_y)**2)
            # Check if landmark is within the field of view and within the maximum distance
            if distance <= self.Camera_maxDist:
                angle_to_landmark = -math.atan2(landmark_y - my_y, landmark_x - my_x)
                angle_difference = abs(math.degrees(self.theta - angle_to_landmark))
                if angle_difference <= self.Camera_fieldView / 2:
                    indices_in_sight.append(i)
        return indices_in_sight
    
    # Method to get the current position of the TurtleBot
    def get_position(self):
        return self.x, self.y

    # Method to get the current orientation of the TurtleBot
    def get_orientation(self):
        return self.theta
    
    # Method to get the current odometry readings
    def get_odometry(self):
        return [self.odometry_left, self.odometry_right]

    # Method to simulate data collection by the TurtleBot from landmarks
    def collect_data(self, landmarks):
        """
        Simulate data collection from landmarks by the TurtleBot.
        Args:
            landmarks: List of landmark positions.
        Returns:
            data_collected: List of collected data with distance and angle to each landmark.
        """
        data_collected = []
        
        for i in random.sample(range(0, len(landmarks))):
            landmark = landmarks[i]
            landmark_x, landmark_y = landmark
            
            # Distance calculation
            distance = euclidean_distance((self.x, self.y), (landmark_x, landmark_y))
            noise_distance = gauss_noise(distance, 0.05)
            if (distance + noise_distance) > 0:
                distance += noise_distance
            
            # Angle calculation
            angle_to_landmark = -math.atan2(landmark_y - self.y, landmark_x - self.x)
            angle_difference = abs(math.degrees(self.theta - angle_to_landmark))
            data_collected.append([distance, angle_difference])
            
        return data_collected
