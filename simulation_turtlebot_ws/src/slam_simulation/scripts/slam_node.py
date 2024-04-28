#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
from particle_filter import Particle

class SimpleSLAM:
    def __init__(self):
        rospy.init_node('simple_slam_node', anonymous=True)
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.num_particles = 50
        self.particles = [Particle() for _ in range(self.num_particles)] #list of particles

        # Map (initially empty)
        self.map = np.zeros((100, 100))  # Example fixed-size grid map
    
        
    def lidar_callback(self, data): # Process lidar data to update the map and robot pose
        rospy.loginfo("Lidar data received")
        ranges = np.array(data.ranges) # Array of range measurements (in meters)
        angle_min = data.angle_min
        angle_increment = data.angle_increment
        
        for particle in self.particles:
            for idx, range_measurement in enumerate(ranges): 
                if range_measurement == float('Inf') or range_measurement == 0.0:
                    # Skip invalid or out-of-range measurements
                    continue

                # Calculate the angle of this measurement
                angle = angle_min + idx * angle_increment

                # Convert polar coordinates (angle, distance) to Cartesian coordinates (x, y)
                x = range_measurement * np.cos(angle + particle.pose[2]) + particle.pose[0]
                y = range_measurement * np.sin(angle + particle.pose[2]) + particle.pose[1]

                # Transform (x, y) into map coordinates
                map_x = int(round((x - self.map_origin_x) / self.map_resolution))
                map_y = int(round((y - self.map_origin_y) / self.map_resolution))

                # Check if the calculated coordinates are within the bounds of the map
                if 0 <= map_x < self.map_width and 0 <= map_y < self.map_height:
                    # Update the map: increment the cell value to indicate an obstacle
                    particle.map[map_y, map_x] += 1

            # Compute weight for this particle
            # Simplified example: Compare expected vs. measured ranges
            weight = self.compute_weight(particle, ranges, angle_min, angle_increment)
            particle.weight = weight

        rospy.loginfo("Particle maps and weights updated")
            
    def resample_particles(self):
        weights = np.array([self.calculate_weight(particle) for particle in self.particles])
        probabilities = weights / np.sum(weights)
        self.particles = np.random.choice(self.particles, size=self.num_particles, p=probabilities)

    def odom_callback(self, msg):
        # Motion model: update each particle's pose based on odometry
        for particle in self.particles:
            # Extract the velocity and angular velocity from the odometry message
            linear_velocity = msg.twist.twist.linear.x  # Forward velocity
            angular_velocity = msg.twist.twist.angular.z  # Angular velocity

            # Time delta: Estimate or calculate based on the rate of odometry messages
            delta_t = 0.1  # Assuming odometry updates every 0.1 seconds

            # Generate random noise for motion model
            # Assuming normal distribution for noise with standard deviations
            noise_lin = np.random.normal(0, 0.02)  # Noise for linear motion
            noise_ang = np.random.normal(0, 0.01)  # Noise for angular motion

            # Predict the new pose based on the current pose and motion
            # Update the orientation
            theta = particle.pose[2] + (angular_velocity + noise_ang) * delta_t
            # Ensure theta is between -pi and pi
            theta = (theta + np.pi) % (2 * np.pi) - np.pi

            # Update the position
            dx = (linear_velocity + noise_lin) * np.cos(theta) * delta_t
            dy = (linear_velocity + noise_lin) * np.sin(theta) * delta_t

            # Update particle pose
            particle.pose[0] += dx
            particle.pose[1] += dy
            particle.pose[2] = theta

            # Log updated pose for debugging
            rospy.loginfo("Updated particle pose: x=%.2f, y=%.2f, theta=%.2f" %
                          (particle.pose[0], particle.pose[1], particle.pose[2]))

    def run(self):
        rospy.Timer(rospy.Duration(1), lambda event: self.resample_particles())
        rospy.spin()


if __name__ == '__main__':
    slam = SimpleSLAM()
    slam.run()
