import math
import numpy as np

class TurtleBot3Waffle:
    def __init__(self, time_interval,Odometry_noise,width_meters, height_meters,turtlebot_radius, x=0, y=0, theta=0):
        self.delta_time = time_interval
        self.Odometry_noise=Odometry_noise
        self.x = x  # x position
        self.y = y  # y position
        self.theta = theta  # orientation (in radians)
        self.wheel_base = 0.287 
        self.wheel_radius = 0.033
        self.maxLinearVel =0.26
        self.maxAngularVel= 1.82 #rad/s
        self.width_meters=width_meters 
        self.height_meters=height_meters
        self.turtlebot_radius = turtlebot_radius

        self.odometry_left = 0
        self.odometry_right = 0
        self.Camera_fieldView = 62.2  #value is in DEGREES
        self.Camera_maxDist = 2 #(meters) maximum distance that camera can detect a aruco with quality

        #parameters for noise in the odometry
        self.mean = 0  # Mean of the Gaussian distribution
        self.std_dev = 0.001  # Standard deviation of the Gaussian distribution


    def move(self, linear_velocity, angular_velocity, delta_time):
        if not delta_time:
            delta_time=self.delta_time
        #checks if velocity is out of range
        if abs(angular_velocity)>self.maxAngularVel:
            angular_velocity = self.maxAngularVel * angular_velocity/(abs(angular_velocity)) #mantains sign of angular velocity
        if abs(linear_velocity)>self.maxLinearVel:
            linear_velocity = self.maxLinearVel * linear_velocity/(abs(linear_velocity)) #mantains sign of velocity

        old_x=self.x
        old_y=self.y
        #kinematics
        v_x = linear_velocity * math.cos(self.theta)
        v_y = linear_velocity * math.sin(self.theta)
        self.x += v_x * delta_time
        self.y -=  v_y * delta_time 
        self.theta +=  angular_velocity * delta_time
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))        # Normalize theta between -pi and pi
        
        #check that turtlebot is inside the screen
        if self.x + self.width_meters/2 > self.width_meters - self.turtlebot_radius:
            self.x=self.width_meters/2 - self.turtlebot_radius
        elif self.x + self.width_meters/2< 0 + self.turtlebot_radius:
            self.x = -self.width_meters/2 + self.turtlebot_radius

        if self.y + self.height_meters/2 > self.height_meters - self.turtlebot_radius:
            self.y=self.height_meters/2- self.turtlebot_radius
        elif self.y + self.height_meters/2< 0 + self.turtlebot_radius:
            self.y = -self.height_meters/2+ self.turtlebot_radius
            
        #odometry
        Translation = math.copysign(math.sqrt((self.x - old_x)**2 + (self.y - old_y)**2), linear_velocity)
        Rotation = angular_velocity * delta_time
        odometry_left = Translation - Rotation*self.wheel_base/2 #this is given in meters
        odometry_right = Translation + Rotation*self.wheel_base/2 #this is given in meters
        gaussian_noise = np.random.normal(self.mean, self.std_dev, 2)
        if self.Odometry_noise==False:
            gaussian_noise=[0,0]
        self.odometry_left += odometry_left + gaussian_noise[0]
        self.odometry_right +=  odometry_right + gaussian_noise[1]

    def check_landmarks(self, landmarks, ):
        indices_in_sight = []
        my_x = self.x + self.width_meters/2
        my_y = self.y +  self.height_meters/2
        for i, landmark in enumerate(landmarks):
            landmark_x, landmark_y = landmark
            #print('Landmark:', i, landmark)
            # Calculate distance between camera and landmark
            distance = math.sqrt(( my_x- landmark_x)**2 + (my_y - landmark_y)**2)
            # Check if landmark is within the field of view and within the maximum distance
            #print(distance, "x, y: ", my_x, my_y)
            
            if distance <= self.Camera_maxDist:
                #print("Passed distance on landmark", i)
                angle_to_landmark = -math.atan2(landmark_y - my_y, landmark_x - my_x)
                angle_difference = abs(math.degrees(self.theta - angle_to_landmark))
                #print("Angle theta ", math.degrees(self.theta)," angle_to_landmark ", math.degrees(angle_to_landmark) )
                if angle_difference <= self.Camera_fieldView / 2:
                    #print("Passed angle on landmark", i)
                    indices_in_sight.append(i)
        return indices_in_sight
    
    def get_position(self):
        return self.x,self.y

    def get_orientation(self):
        return self.theta
    
    def get_odometry(self):
        return self.odometry_left, self.odometry_right


    