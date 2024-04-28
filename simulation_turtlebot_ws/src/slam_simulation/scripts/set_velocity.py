#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist

def set_velocity():
    # Initialize the ROS Node
    rospy.init_node('set_constant_velocity', anonymous=True)

    # Publisher to publish Twist messages on the 'cmd_vel' topic
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    # Set the loop rate (in Hz)
    rate = rospy.Rate(10)  # 10 Hz

    # Create a Twist message instance
    vel_msg = Twist()

    # Set linear velocity in x to 0.1 m/s
    vel_msg.linear.x = 0.1
    vel_msg.linear.y = 0
    vel_msg.linear.z = 0

    # Set angular velocity to 0
    vel_msg.angular.x = 0
    vel_msg.angular.y = 0
    vel_msg.angular.z = 0

    # Keep publishing the velocity at a set rate
    while not rospy.is_shutdown():
        pub.publish(vel_msg)
        rospy.loginfo("Publishing velocity: %s" % vel_msg.linear.x)
        rate.sleep()

if __name__ == '__main__':
    try:
        set_velocity()
    except rospy.ROSInterruptException:
        pass
