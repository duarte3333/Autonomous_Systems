import numpy as np
import math
import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

class OcupancyGridMap:
    def __init__(self, width, height, resolution, frame_id="map"):
        # Initialize parameters for the occupancy grid map
        self.count_until_disapear = 0
        self.width = int(width / resolution) + 1  # Number of cells in width
        self.height = int(height / resolution) + 1  # Number of cells in height
        self.resolution = resolution  # Size of each cell in meters
        self.laser_tf_static = [-0.064, 0]  # Static transformation for the laser sensor
        self.frame_id = frame_id  # Frame ID for the occupancy grid
        
        # Parameters for the inverse sensor model
        self.l_occ = np.log(0.9 / 0.1)  # Log-odds for occupied
        self.l_free = np.log(0.1 / 0.9) / 3  # Log-odds for free
        self.logDecrement = np.log(0.1 / 0.9) / 10  # Log odds decrement for forgetting past measurements
        self.threshold = 100  # Threshold for resetting cells
        
        self.l0 = 0  # Log-odds for unknown (0.5 probability)
        
        # Initialize the occupancy grid with log-odds of zero (equivalent to 0.5 probability)
        self.grid = np.zeros((self.height, self.width))
        self.disappear_count = np.zeros((self.height, self.width))

    # Convert world coordinates to grid coordinates
    def world_to_grid(self, x, y):
        gx = int((x / self.resolution) + (self.width / 2))
        gy = int((y / self.resolution) + (self.height / 2))
        return gx, gy

    # Update the occupancy grid based on laser scan data and robot pose
    def update(self, robot_pose, laser_scan):
        x, y, yaw = robot_pose
        x = -x
        yaw -= np.pi
        
        # Convert robot position to grid coordinates
        robot_gx, robot_gy = self.world_to_grid(x, y)

        angle_min = laser_scan.angle_min
        angle_increment = laser_scan.angle_increment
        ranges = laser_scan.ranges
        self.count_sub = 0
        self.count_add = 0
        
        for i, distance in enumerate(ranges):
            if distance > 0:
                angle = angle_min + i * angle_increment
                angle_rad = angle + yaw
                lx = x + distance * np.cos(angle_rad) + self.laser_tf_static[0]
                ly = y + distance * np.sin(angle_rad) + self.laser_tf_static[1]

                # Convert laser point to grid coordinates
                laser_gx, laser_gy = self.world_to_grid(lx, ly)

                # Update grid cells along the line from robot to laser point
                self.bresenham_update(robot_gx, robot_gy, laser_gx, laser_gy, distance)
        
        # Decrease the confidence of occupied cells over time
        self.grid = np.where(self.grid > 0, self.grid + self.logDecrement, self.grid)
        
        # Reset cells that have been occupied for too long
        self.grid[self.disappear_count > self.threshold] = self.l0
        self.disappear_count[self.disappear_count > self.threshold] = 0

    # Use Bresenham's algorithm to update the grid cells along the line
    def bresenham_update(self, x0, y0, x1, y1, distance):
        def bresenham_line(x0, y0, x1, y1, update_func):
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            x, y = x0, y0
            sx = -1 if x0 > x1 else 1
            sy = -1 if y0 > y1 else 1
            if dx > dy:
                err = dx / 2.0
                while x != x1:
                    update_func(x, y)
                    err -= dy
                    if err < 0:
                        y += sy
                        err += dx
                    x += sx
            else:
                err = dy / 2.0
                while y != y1:
                    update_func(x, y)
                    err -= dx
                    if err < 0:
                        x += sx
                        err += dy
                    y += sy
            update_func(x, y)

        # Update free cells
        def forward_update(x, y):
            self.grid[y, x] = self.grid[y, x] + self.l_free
            self.count_sub += 1

        # Track cells to reset if occupied for too long
        def reverse_update(x, y):
            if self.grid[y, x] != 0:
                self.disappear_count[y, x] += 1

        # Perform the original Bresenham algorithm for the lidar measurement
        bresenham_line(x0, y0, x1, y1, forward_update)

        # Update the endpoint (occupied cell)
        if 0 <= x1 < self.width and 0 <= y1 < self.height:
            self.grid[y1, x1] = self.grid[y1, x1] + self.l_occ
            self.count_add += 1

        # Calculate the point behind the lidar measurement
        x2 = int(x1 + (x1 - x0))
        y2 = int(y1 + (y1 - y0))
        y2_checked = y2 if y2 < self.height else self.height - 1
        y2_checked = y2 if y2 > 0 else 0
        x2_checked = x2 if x2 < self.width else self.width - 1
        x2_checked = x2 if x2 > 0 else 0

        # Perform the reverse Bresenham algorithm for the points behind the lidar measurement
        bresenham_line(x1 + int((x1 - x0) * 0.15), y1 + int((y1 - y0) * 0.15), x2_checked, y2_checked, reverse_update)
    
    # Convert log-odds to probabilities
    def get_probabilities(self):
        return 1 - 1 / (1 + np.exp(self.grid))
    
    # Get normalized probabilities (same as get_probabilities in this case)
    def get_normalized_probabilities(self):
        return self.get_probabilities()

    # Get the occupancy grid in binary form (0 = free, 1 = occupied)
    def get_occupancy_grid(self):
        return (self.get_probabilities() > 0.3).astype(np.int8)

    # Publish the occupancy grid as a nav_msgs/OccupancyGrid message
    def publish_grid(self, Occu_grid_pub):
        grid_msg = OccupancyGrid()
        grid_msg.header = Header()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = self.frame_id
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        grid_msg.info.origin.position.x = -self.width / 2 * self.resolution
        grid_msg.info.origin.position.y = -self.height / 2 * self.resolution
        grid_msg.info.origin.position.z = 0
        grid_msg.info.origin.orientation.w = 1.0

        # Convert the grid probabilities to a flat list of 8-bit integers
        grid_msg.data = (self.get_probabilities().flatten() * 100).astype(np.int8).tolist()
        Occu_grid_pub.publish(grid_msg)
