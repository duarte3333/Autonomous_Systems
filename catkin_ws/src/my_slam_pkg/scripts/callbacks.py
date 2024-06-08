import copy

# Callback function for odometry data
def odom_callback(self, odom_data):
    with self.lock:  # Ensure thread-safe operation with a lock
        # Extract position and orientation from odometry data
        x = odom_data.pose.pose.position.x
        y = odom_data.pose.pose.position.y
        xq = odom_data.pose.pose.orientation.x
        yq = odom_data.pose.pose.orientation.y
        zq = odom_data.pose.pose.orientation.z
        wq = odom_data.pose.pose.orientation.w
        quater = [xq, yq, zq, wq]
        
        # Update the odometry information
        self.odom = [x, y, quater]
        
        # Initialize reference (tara) position on the first callback
        if self.count == 0:
            self.tara = copy.deepcopy(self.odom)
            self.count += 1

        # Adjust odometry based on the initial reference position
        self.odom[0] -= self.tara[0]
        self.odom[1] -= self.tara[1]
        
        # Update SLAM with the new odometry information
        self.my_slam.update_odometry(self.odom)

# Callback function for laser scan data
def lidar_callback(self, laser_scan):
    with self.lock:  # Ensure thread-safe operation with a lock
        # Update SLAM with the new laser scan information
        self.my_slam.update_laser_scan(laser_scan)
