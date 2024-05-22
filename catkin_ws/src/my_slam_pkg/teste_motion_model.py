#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Polygon
import numpy as np
import tf.transformations as tf_trans


# Initialize global variables
x_data = []
y_data = []
theta = 0

# Callback function to process odom messages
def odom_callback(msg):
    global theta

    # Extract position
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    
    # Extract orientation in quaternion and convert to euler (yaw)
    orientation_q = msg.pose.pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, theta) = tf_trans.euler_from_quaternion(orientation_list)

    # Append position to data lists
    x_data.append(x)
    y_data.append(y)

# Initialize ROS node
rospy.init_node('path_tracer')

# Subscribe to the /odom topic
rospy.Subscriber('/odom', Odometry, odom_callback)

# Set up the plot
fig, ax = plt.subplots()
path_line, = ax.plot([], [], 'b-')
robot_circle = Circle((0, 0), 0.1, fc='r')
robot_dir = Polygon([[0, 0], [0.1, 0], [0.05, 0.1]], closed=True, fc='g')
ax.add_patch(robot_circle)
ax.add_patch(robot_dir)
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_aspect('equal')

def init():
    path_line.set_data([], [])
    robot_circle.center = (0, 0)
    robot_dir.xy = [[0, 0], [0.1, 0], [0.05, 0.1]]
    return path_line, robot_circle, robot_dir

def animate(i):
    if len(x_data) == 0:
        return path_line, robot_circle, robot_dir

    path_line.set_data(x_data, y_data)
    robot_circle.center = (x_data[-1], y_data[-1])
    
    # Update the direction indicator
    x, y = x_data[-1], y_data[-1]
    dx = 0.1 * np.cos(theta)
    dy = 0.1 * np.sin(theta)
    robot_dir.xy = [[x, y], [x + dx, y + dy], [x + 0.05 * np.cos(theta + np.pi/2), y + 0.05 * np.sin(theta + np.pi/2)]]

    return path_line, robot_circle, robot_dir

ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=100, blit=True)

plt.show()
rospy.spin()
