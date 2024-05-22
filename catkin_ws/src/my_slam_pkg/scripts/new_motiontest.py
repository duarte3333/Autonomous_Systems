import rospy
import tf
from nav_msgs.msg import Odometry
from math import atan2, sqrt, cos, sin, pi
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to sample noise with clipping
def sample(b):
    # Use a reasonable clip value to avoid numerical overflow
    clip_value = 1e-10
    return random.gauss(0, b) if b > clip_value else 0

# Normalize angle to be within [-pi, pi]
def normalize_angle(angle):
    while angle > pi:
        angle -= 2.0 * pi
    while angle < -pi:
        angle += 2.0 * pi
    return angle

# Odometry callback function
def odom_callback(data):
    global prev_x, prev_y, prev_theta
    global x, y, theta
    global trajectory_x, trajectory_y
    global theta_prime
    pose = data.pose.pose
    x = pose.position.x
    y = pose.position.y
    
    # Convert quaternion to euler
    orientation_q = pose.orientation
    orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
    (roll, pitch, theta) = tf.transformations.euler_from_quaternion(orientation_list)
    
    theta = normalize_angle(theta)
    print('YAW IS', theta)
    if prev_x is None:
        prev_x = x
        prev_y = y
        prev_theta = theta
        return

    # Algorithm parameters
    alpha1=0.00001
    alpha2=0.00001
    alpha3=0.00001
    alpha4=0.00001

    # Step 2
    delta_rot1 = normalize_angle(atan2(y - prev_y, x - prev_x) - prev_theta)

    # Step 3
    delta_trans = sqrt((prev_x - x)**2 + (prev_y - y)**2)

    # Step 4
    delta_rot2 = normalize_angle(theta - prev_theta - delta_rot1)

    # Step 5
    delta_rot1_hat = delta_rot1 - sample(alpha1 * delta_rot1**2 + alpha2 * delta_trans**2)

    # Step 6
    delta_trans_hat = delta_trans - sample(alpha3 * delta_trans**2 + alpha4 * delta_rot1**2 + alpha4 * delta_rot2**2)

    # Step 7
    delta_rot2_hat = delta_rot2 - sample(alpha1 * delta_rot2**2 + alpha2 * delta_trans**2)

    # Step 8
    x_prime = prev_x + delta_trans_hat * cos(prev_theta + delta_rot1_hat)

    # Step 9
    y_prime = prev_y + delta_trans_hat * sin(prev_theta + delta_rot1_hat)

    # Step 10
    theta_prime = normalize_angle(prev_theta + delta_rot1_hat + delta_rot2_hat)

    # Update previous state
    prev_x = x_prime
    prev_y = y_prime
    prev_theta = theta_prime

    # Append the new state to the trajectory
    trajectory_x.append(x_prime)
    trajectory_y.append(y_prime)

    # Print or publish the updated state as needed
    #rospy.loginfo(f"Updated state: x = {x_prime}, y = {y_prime}, theta = {theta_prime}")

# Initialize node
rospy.init_node('motion_model_odometry')

# Initialize global variables
prev_x = None
prev_y = None
prev_theta = None

x = None
y = None
theta = None

trajectory_x = []
trajectory_y = []

# Subscribe to /odom topic
rospy.Subscriber('/odom', Odometry, odom_callback)

# Plotting setup
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-')  # Line object for trajectory

robot_circle = patches.Circle((0, 0), 0.1, fc='blue')
robot_triangle = patches.Polygon([[0, 0], [0, 0], [0, 0]], closed=True, fc='blue')

ax.add_patch(robot_circle)
ax.add_patch(robot_triangle)

ax.set_xlim(-10, 10)  # Set the x-axis limits
ax.set_ylim(-10, 10)  # Set the y-axis limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Robot Trajectory')

# Function to update the plot
def update_plot():
    if trajectory_x and trajectory_y:  # Check if lists are not empty
        line.set_data(trajectory_x, trajectory_y)
        ax.set_xlim(min(trajectory_x) - 1, max(trajectory_x) + 1)
        ax.set_ylim(min(trajectory_y) - 1, max(trajectory_y) + 1)
        
        # Update the circle position
        robot_circle.center = (trajectory_x[-1], trajectory_y[-1])
        
        # Update the triangle position and orientation
        tri_base = 0.1  # Base width of the triangle
        tri_height = 0.2  # Height of the triangle
        triangle_points = [
            [trajectory_x[-1] + tri_height * cos(theta_prime), trajectory_y[-1] + tri_height * sin(theta_prime)],
            [trajectory_x[-1] + tri_base * cos(theta_prime + pi / 2), trajectory_y[-1] + tri_base * sin(theta_prime + pi / 2)],
            [trajectory_x[-1] + tri_base * cos(theta_prime - pi / 2), trajectory_y[-1] + tri_base * sin(theta_prime - pi / 2)]
        ]
        robot_triangle.set_xy(triangle_points)
        
    fig.canvas.draw()
    fig.canvas.flush_events()

# Main loop
rate = rospy.Rate(10)  # 10 Hz
while not rospy.is_shutdown():
    update_plot()
    rate.sleep()
