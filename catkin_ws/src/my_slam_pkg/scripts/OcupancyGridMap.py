
import numpy as np
import math
import rospy
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

class OcupancyGridMap:
    def __init__(self, width, height, resolution, frame_id="map"):
        
        
        self.width = int(width/resolution)+1 #number os cells
        self.height = int(height/resolution)+1 #nmumber os cells
        self.resolution = resolution #meters of each cell

        self.frame_id = frame_id
        # Initialize the occupancy grid with log-odds of zero (equiv. to 0.5 probability)
        self.grid = np.zeros((self.height, self.width), dtype=np.float32)
        # Parameters for the inverse sensor model
        self.l_occ = np.log(0.7 / 0.3)   # Log-odds for occupied
        self.l_free = np.log(0.3 / 0.7)  # Log-odds for free
        
        self.l_occ = np.log(9)  # Log odds for occupied
        self.l_free = np.log(1/9)  # Log odds for free
        

        self.l0 = 0                      # Log-odds for unknown (0.5 probability)

        

    def world_to_grid(self, x, y):
        """Convert world coordinates to grid coordinates"""
        gx = int((x / self.resolution) + (self.width / 2))
        gy = int((y / self.resolution) + (self.height / 2))
        return gx, gy

    def update(self, robot_pose, laser_scan):
        """Update the occupancy grid based on laser scan data and robot pose"""
        x, y, yaw = robot_pose

        # Convert robot position to grid coordinates
        robot_gx, robot_gy = self.world_to_grid(x, y)

        angle_min = laser_scan.angle_min
        angle_increment = laser_scan.angle_increment
        ranges = laser_scan.ranges

        for i, distance in enumerate(ranges):
            if distance > 0:
                angle = angle_min + i * angle_increment
                angle_rad = angle + yaw
                lx = x + distance * np.cos(angle_rad)
                ly = y + distance * np.sin(angle_rad)

                # Convert laser point to grid coordinates
                laser_gx, laser_gy = self.world_to_grid(lx, ly)

                # Update grid cells along the line from robot to laser point
                self.bresenham_update(robot_gx, robot_gy, laser_gx, laser_gy, distance)


    def bresenham_update(self, x0, y0, x1, y1, distance):
        """Use Bresenham's algorithm to update the grid cells along the line"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        while True: 
            self.grid[y0, x0] += self.l_free  # Update free cells
            if x0 == x1 and y0 == y1:
                break
            if dx == 0 and dy == 0:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy

        # Update the endpoint (occupied cell)
        if 0 <= x1 < self.width and 0 <= y1 < self.height:
              # Only update if within max sensor range
            self.grid[y1, x1] += self.l_occ


#self.grid[y0, x0] += self.l_free  # Update free cells
#IndexError: index 50 is out of bounds for axis 0 with size 10

    def get_probabilities(self):
        """Convert log-odds to probabilities"""
        return 1 - 1 / (1 + np.exp(self.grid))

    def get_occupancy_grid(self):
        """Get the occupancy grid in binary form (0 = free, 1 = occupied)"""
        return (self.get_probabilities() > 0.5).astype(np.int8)

    def publish_grid(self,Occu_grid_pub ):
        """Publish the occupancy grid as a nav_msgs/OccupancyGrid message"""
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

        grid_msg.data = (self.get_probabilities().flatten() * 100).astype(np.int8).tolist()
        Occu_grid_pub.publish(grid_msg)
