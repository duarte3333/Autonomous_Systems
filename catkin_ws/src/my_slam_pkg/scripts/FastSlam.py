import pygame
import math 
import numpy as np
import copy
from aux_slam import resample, normalize_angle
from Particle import Particle
import tf.transformations
import rospy
from visualization_msgs.msg import Marker, MarkerArray

class FastSlam:
    def publish_landmarks(self):
        marker_array = MarkerArray()
        for landmark_id, landmark in self.particles[self.best_particle_ID].landmarks.items():
            landmark_x , landmark_y = landmark.x, landmark.y

            # Create a marker for the landmark
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "landmarks"
            marker.id = int(landmark_id)
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = -landmark_x
            marker.pose.position.y = landmark_y
            marker.pose.position.z = 0  # Assuming 2D landmarks
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            marker.color.a = 1.0  # Don't forget to set the alpha!
            marker.color.r = 0.0
            marker.color.g = 255.0
            marker.color.b = 0.0

            # Add the marker to the array
            marker_array.markers.append(marker)

            # Create a marker for the text
            text_marker = Marker()
            text_marker.header.frame_id = "map"
            text_marker.header.stamp = rospy.Time.now()
            text_marker.ns = "landmark_text"
            text_marker.id = int(landmark_id) + 1000  # Ensure unique ID for text
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position.x = -landmark_x
            text_marker.pose.position.y = landmark_y
            text_marker.pose.position.z = 0.5  # Slightly above the landmark
            text_marker.pose.orientation.x = 0.0
            text_marker.pose.orientation.y = 0.0
            text_marker.pose.orientation.z = 0.0
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = 0.4  # Text height
            text_marker.color.a = 1.0  # Don't forget to set the alpha!
            text_marker.color.r = 1.0
            text_marker.color.g = 255.0
            text_marker.color.b = 1.0
            text_marker.text = f"id: {landmark_id}"

            # Add the text marker to the array
            marker_array.markers.append(text_marker)

        # Publish the marker array
        self.landmark_pub.publish(marker_array)
    def __init__(self,only_slam_window, window_size_pixel, sample_rate, size_m,central_bar_width,OG_map_options,Occu_grid_pub, turtlebot_L,num_particles=50 , screen=None, resample_method="low variance",std_dev_motion = 0.5):
        
        self.SCREEN_WIDTH = window_size_pixel
        self.SCREEN_HEIGHT = window_size_pixel
        self.only_slam_window=only_slam_window
        self.central_bar_width=central_bar_width

        if screen==None:
            pygame.init()
            pygame.display.init()
            screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("TurtleBot3 Slam predictions")
            self.left_coordinate = 0
            self.right_coordinate=self.SCREEN_WIDTH
            
        elif only_slam_window:
            self.left_coordinate = 0
            self.right_coordinate=self.SCREEN_WIDTH
        else:
            self.left_coordinate = central_bar_width + self.SCREEN_WIDTH
            self.right_coordinate=self.SCREEN_WIDTH+self.left_coordinate+self.central_bar_width
        
        self.std_dev_motion = std_dev_motion
        self.resample_method=resample_method
        # Define colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 160, 0)
        self.BLUE=(10,10,255)
        self.RED=(170,0,0)

        # Set up the screen
        self.screen = screen      
        self.width_meters=size_m 
        self.height_meters=size_m
        self.turtlebot_radius=0.105
        self.turtlebot_L=turtlebot_L
 
        self.turtlebot_radius_pixel =self.turtlebot_radius * self.SCREEN_WIDTH/self.width_meters #from turtlebot website

        self.old_odometry = [0.0,0.0]
        self.old_yaw = 0
        self.num_particles = num_particles
        self.best_particle_ID=-1
        self.OG_mapoptions = OG_map_options
        self.Occu_grid_pub=Occu_grid_pub
        self.particles = self.initialize_particles()
        self.landmark_pub = rospy.Publisher('/landmarks', MarkerArray, queue_size=10)
        rospy.init_node('aruco_slam')  # Initialize the ROS node
        self.update_screen()
        #initialize FastSlam variables
        return
    
    def get_best_particle(self):
        return self.particles[self.best_particle_ID]
    
    def initialize_particles(self, landmarks={}):
        particles = []
        for _ in range(self.num_particles):
            # Initialize each particle with a random pose and empty landmarks
            x = 0#np.random.uniform(0, self.width_meters)
            y = 0#np.random.uniform(0, self.height_meters)
            theta = 0#np.random.uniform(0, 2 * np.pi)
            pose = np.array([x, y, theta])
            particles.append(Particle(pose,self.num_particles, self.turtlebot_L,self.OG_mapoptions,self.std_dev_motion ))

        return particles
    
    def update_laser_scan(self,scan_msg):
        for particle in self.particles:
            particle.updateMap(scan_msg)
        
        best_particle = self.particles[self.best_particle_ID]
        best_particle.OG_map.publish_grid(self.Occu_grid_pub)

    def update_odometry(self,odometry):
        
        quaternion = [odometry[2][0],odometry[2][1],odometry[2][2],odometry[2][3]]
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(quaternion)
        yaw = normalize_angle(yaw)
  
        delta_dist = math.sqrt((odometry[0]-self.old_odometry[0])**2+(odometry[1]-self.old_odometry[1])**2)
        delta_rot1 = normalize_angle(math.atan2(odometry[1]-self.old_odometry[1],odometry[0]-self.old_odometry[0])- self.old_yaw)
        delta_rot2 = normalize_angle(yaw - self.old_yaw- delta_rot1)

        # Update each particle with motion and observation models
        for particle in self.particles:
            # Motion update
            particle.motion_model([delta_dist, delta_rot1, delta_rot2])
        self.old_odometry= copy.deepcopy(odometry)
        self.old_yaw = copy.deepcopy(yaw)
        self.update_screen()



    def compute_slam(self, landmarks_in_sight):
        #compute and display FastSlam
        #self.update_odometry(odometry)
        # Landmark update
        weights_here=[]
        for landmark in landmarks_in_sight:
            landmark_dist, landmark_bearing_angle, landmark_id = landmark
            x,y,theta= self.particles[0].pose
            weights_here=[]
            for particle in self.particles:
                particle.handle_landmark(landmark_dist, math.radians(landmark_bearing_angle), landmark_id)
                weights_here.append(particle.weight)    
                #print particle with more weight 
                #print('max weight', max(weights_here))  
            
       
        self.particles , self.best_particle_ID = resample(self.particles, self.num_particles, self.resample_method, self.best_particle_ID)
        #use latest estimation to update_screen
        self.update_screen(landmarks_in_sight)


    def update_screen(self, landmarks_in_sight=None):
        if self.best_particle_ID==-1:
           # x,y,theta= 0,0,0
           self.best_particle_ID=np.random.randint(len(self.particles))

        x,y,theta= self.particles[self.best_particle_ID].pose
        #y=-y
        # Calculate the vertices of the triangle for orientation
        turtlebot_pos= (int((x) * self.SCREEN_WIDTH/self.width_meters +self.left_coordinate + self.SCREEN_WIDTH/2), int((y) *self.SCREEN_HEIGHT/self.height_meters + self.SCREEN_HEIGHT/2)) #window should display a 5x5 m^2 area
        triangle_length = 0.8*self.turtlebot_radius_pixel
        triangle_tip_x = turtlebot_pos[0] + triangle_length * math.cos(theta)
        triangle_tip_y = turtlebot_pos[1] - triangle_length * math.sin(theta)
        triangle_left_x = turtlebot_pos[0] + triangle_length * math.cos(theta + 5 * math.pi / 6) 
        triangle_left_y = turtlebot_pos[1] - triangle_length * math.sin(theta + 5 * math.pi / 6) 
        triangle_right_x = turtlebot_pos[0] + triangle_length * math.cos(theta - 5 * math.pi / 6)
        triangle_right_y = turtlebot_pos[1] - triangle_length * math.sin(theta - 5 * math.pi / 6)

        # Draw the triangle
        triangle_points = [(triangle_tip_x, triangle_tip_y), (triangle_left_x, triangle_left_y), (triangle_right_x, triangle_right_y)]
        #fill half of the screen
        half_screen_rect = pygame.Rect(self.left_coordinate, 0, self.right_coordinate, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.WHITE, half_screen_rect)

        pygame.draw.circle(self.screen, self.GREEN, turtlebot_pos , self.turtlebot_radius_pixel)
        pygame.draw.polygon(self.screen, self.BLUE, triangle_points)

        #draw current particles
        for particle in self.particles:
            particle_x , particle_y, _ = particle.pose
            #particle_y = -particle_y
            pygame.draw.circle(self.screen, self.RED, (int((particle_x ) * self.SCREEN_WIDTH/self.width_meters + self.left_coordinate + self.SCREEN_WIDTH/2), int((particle_y)* self.SCREEN_HEIGHT/self.height_meters+ self.SCREEN_HEIGHT/2)), 3)

        for landmark_id, landmark in self.particles[self.best_particle_ID].landmarks.items():
            landmark_x , landmark_y = landmark.x, landmark.y
            
            #print('landmark_id',landmark_id, ' pose: ', landmark_x ,' ,', landmark_y)
            pygame.draw.circle(self.screen, self.BLACK, (int(landmark_x* self.SCREEN_WIDTH/self.width_meters + self.left_coordinate + self.SCREEN_WIDTH/2), int(landmark_y* self.SCREEN_HEIGHT/self.height_meters+ self.SCREEN_HEIGHT/2)), 5)
            import rospy
            #self.image_sub = rospy.Publisher("/raspicam_node/image/compressed", CompressedImage, self.image_callback_wrapper)
            
            font = pygame.font.Font(None, 30)  
            text_surface = font.render("id:"+str(landmark_id), True, self.BLACK)  # Render text surface
            text_rect = text_surface.get_rect(center=(int(landmark_x * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH/2), int(landmark_y * self.SCREEN_HEIGHT / self.height_meters+ self.SCREEN_HEIGHT/2)- 15))  # Position text surface above the circle
            self.screen.blit(text_surface, text_rect)  # Blit text surface onto the screen
        
        if self.only_slam_window:
            pygame.display.flip()

    def get_best_trajectory(self):
        return self.particles[self.best_particle_ID].trajectory
    
    def get_BestLandmarks(self):
        first=True
        for landmark_id, landmark in self.particles[self.best_particle_ID].landmarks.items():
            if first:
                first=False
                slam_landmarks = np.array([int(landmark_id),float(landmark.mu[0][0]),float(landmark.mu[1][0])])
            else:
                this_landmark= np.array([int(landmark_id),float(landmark.mu[0][0]),float(landmark.mu[1][0])])
             
                slam_landmarks = np.vstack((slam_landmarks, this_landmark))
        slam_landmarks=slam_landmarks.T
        sorted_indices = np.argsort(slam_landmarks[0,:])
        return slam_landmarks[:,sorted_indices] #sort array with the IDS



