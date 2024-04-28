#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

from robot_pose import RobotPose
from laser_scan import LaserScan
from geometry_msgs.msg import Twist
from particle_filter import Particle


class CustomizedSLAM():
    def __init__(self, map_size=(40, 40), step_size=0.2):
        self.resampled_times = 0
        self.map_size = map_size
        self.step_size = step_size
        self.current_pose = RobotPose()
        self.current_scan = LaserScan()
        #self.sensor_model
        self.got_scan = False
        self.first_odom = True
        self.start = False
        
        rospy.init_node('simple_slam_node', anonymous=True)
        self.lidar_sub = rospy.Subscriber('/base_scan', LaserScan, self.lidar_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # Map (initially empty)
        self.map = np.zeros((100, 100))  # Example fixed-size grid map
    
    def lidar_callback(self, data):
        self.current_scan = data
        self.got_scan = True
        
    def odom_callback(self, data):
        #quarernion has one real part and three imaginary parts
        orientation_q = data.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list) #yaw is rotation about z axis
        self.current_pose.x = data.pose.pose.position.x
        self.current_pose.y = data.pose.pose.position.y
        self.current_pose.theta = yaw


    def run(self, event = None):
        pass
        # while not rospy.is_shutdown():
        #     if self.got_scan and self.first_odom and self.start:
        #         #self.current_scan = self.new_scan
        #         self.map = self.sensor_model.update_map(self.current_scan, self.current_pose, self.map)
        #         try:
        #         	gridToNpyFile(self.map, self.current_pose, "./maps", "map" + str(Mapper.iteration))
        #         	#gridToNpyFile(self.map, self.current_pose, "./maps", "map_final")
        #         except:
        #         	pass
        #         Mapper.iteration += 1
        #         rospy.loginfo(str(Mapper.iteration))
        #         self.got_scan = False


if __name__ == '__main__':
    rospy.loginfo("Starting Simple SLAM Node")
    try: 
        my_map = CustomizedSLAM()
    except Exception as er:
        rospy.logerr(er)
    finally:
        my_map.run()
