import copy

def odom_callback(self, odom_data):
    with self.lock:
        x = odom_data.pose.pose.position.x
        y = odom_data.pose.pose.position.y
        xq = odom_data.pose.pose.orientation.x
        yq = odom_data.pose.pose.orientation.y
        zq = odom_data.pose.pose.orientation.z
        wq = odom_data.pose.pose.orientation.w
        quater = [xq,yq,zq,wq]
        self.odom = [x,y,quater]
        if self.count == 0:
            self.tara = copy.deepcopy(self.odom)
            self.count +=1

        self.odom[0] -= self.tara[0]
        self.odom[1] -= self.tara[1]
        self.my_slam.update_odometry(self.odom)