#!/usr/bin/env python3

import rospy

def simple_node():
    rospy.init_node('simple_test_node', anonymous=True)
    rospy.loginfo("Simple test node is running")
    rospy.spin()

if __name__ == '__main__':
    try:
        simple_node()
    except rospy.ROSInterruptException:
        pass
