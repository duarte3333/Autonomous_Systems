#!/usr/bin/env python3
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from sensor_msgs.msg import LaserScan

def cmd_vel_callback(data):
    rospy.loginfo("Received velocity command: %s", data)

def reset_callback(data):
    rospy.loginfo("Reset command received.")

def turtlebot_interface():
    rospy.init_node('turtlebot_interface', anonymous=True)

    # Subscribers
    rospy.Subscriber("/cmd_vel", Twist, cmd_vel_callback)
    rospy.Subscriber("/reset", Empty, reset_callback)

    # Publishers
    odom_pub = rospy.Publisher("/odom", Odometry, queue_size=10)
    tf_pub = rospy.Publisher("/tf", TFMessage, queue_size=10)
    scan_pub = rospy.Publisher("/scan", LaserScan, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz

    while not rospy.is_shutdown():
        # Create dummy Odometry data
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "odom"
        odom_msg.child_frame_id = "base_link"
        odom_pub.publish(odom_msg)

        # Create dummy TF data
        tf_msg = TFMessage()
        tf_pub.publish(tf_msg)

        # Create dummy LaserScan data
        scan_msg = LaserScan()
        scan_msg.header.stamp = rospy.Time.now()
        scan_msg.header.frame_id = "laser"
        scan_pub.publish(scan_msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        turtlebot_interface()
    except rospy.ROSInterruptException:
        pass
