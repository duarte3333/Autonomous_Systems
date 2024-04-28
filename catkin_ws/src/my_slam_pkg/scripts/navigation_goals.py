#!/usr/bin/env python
import rospy
from actionlib import SimpleActionClient
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

def move_to_goal(x, y):
    client = SimpleActionClient('move_base', MoveBaseAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    goal.target_pose.pose.orientation.w = 1.0  # facing straight ahead

    rospy.loginfo("Sending goal to ({}, {})".format(x, y))
    client.send_goal(goal)
    client.wait_for_result()
    if client.get_state() == 3:
        rospy.loginfo("Goal reached!")
    else:
        rospy.loginfo("Failed to reach the goal.")

if __name__ == '__main__':
    rospy.init_node('navigate_to_point')
    try:
        move_to_goal(1.0, 1.0)  # You can change these coordinates as needed
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation interrupted.")
