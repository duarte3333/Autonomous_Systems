class LaserScan:
    def __init__(self, scan_message=None):
        # Initialize from scan_message if provided, otherwise use defaults
        if scan_message is not None:
            self.angle_min = scan_message.angle_min
            self.angle_max = scan_message.angle_max
            self.angle_increment = scan_message.angle_increment
            self.time_increment = scan_message.time_increment
            self.scan_time = scan_message.scan_time
            self.range_min = scan_message.range_min
            self.range_max = scan_message.range_max
            self.ranges = scan_message.ranges
            self.intensities = scan_message.intensities
        else:
            # Default values if no scan_message is provided
            self.angle_min = 0
            self.angle_max = 0
            self.angle_increment = 0
            self.time_increment = 0
            self.scan_time = 0
            self.range_min = 0
            self.range_max = 0
            self.ranges = []
            self.intensities = []
