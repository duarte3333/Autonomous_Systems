#!/usr/bin/env python
import rospy
from sensor_msgs.msg import LaserScan
import numpy as np
import tf

class EKFTurtlebotSLAM:
    def __init__(self):
        rospy.init_node('ekf_slam')
        
        # Subscriber to the TurtleBot3's laser scan topic
        self.scan_subscriber = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # Initialize EKF components
        self.state_estimation = np.zeros((3, 1))  # [x, y, theta]
        self.map = None  # This will eventually hold your map data

    def scan_callback(self, data):
        # Here you would process the scan data
        # You could use the scan data to update your map and pose estimate
        pass

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    slam = EKFTurtlebotSLAM()
    slam.run()
