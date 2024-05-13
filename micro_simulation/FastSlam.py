import pygame
import math 
import numpy as np
import copy
from aux_slam import resample
from Particule import Particle

class FastSlam:
    def __init__(self, window_size_pixel, sample_rate, size_m,central_bar_width, turtlebot_L,screen,num_particles=50 , resample_method="low variance",std_dev_motion = 0.5):
        
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
        self.central_bar_width=central_bar_width
        self.SCREEN_WIDTH = window_size_pixel
        self.SCREEN_HEIGHT = window_size_pixel
        self.left_coordinate = central_bar_width + self.SCREEN_WIDTH
        self.right_coordinate=self.SCREEN_WIDTH+self.left_coordinate+self.central_bar_width
        self.width_meters=size_m 
        self.height_meters=size_m
        self.turtlebot_radius=0.105
        self.turtlebot_L=turtlebot_L
 
        self.turtlebot_radius_pixel =self.turtlebot_radius * self.SCREEN_WIDTH/self.width_meters #from turtlebot website

        self.old_odometry = [0,0]

        self.num_particles = num_particles
        self.particles = self.initialize_particles()
        self.best_particle_ID=-1

        self.update_screen()

        #initialize FastSlam variables



        return
    
    def initialize_particles(self, landmarks={}):
        particles = []
        for _ in range(self.num_particles):
            # Initialize each particle with a random pose and empty landmarks
            x = 0#np.random.uniform(0, self.width_meters)
            y = 0#np.random.uniform(0, self.height_meters)
            theta = 0#np.random.uniform(0, 2 * np.pi)
            pose = np.array([x, y, theta])
            particles.append(Particle(pose, self.turtlebot_L,self.std_dev_motion ))
        return particles
    

    def update_odometry(self,odometry):
        deltaRight=odometry[0]-self.old_odometry[0]
        deltaLeft=odometry[1]-self.old_odometry[1]
        # Update each particle with motion and observation models
        for particle in self.particles:
            # Motion update
            particle.motion_model([deltaLeft, deltaRight])
        self.old_odometry= copy.deepcopy(odometry)
        self.update_screen()



    def compute_slam(self, odometry, landmarks_in_sight):
        #compute and display FastSlam
        self.update_odometry(odometry)
        # Landmark update
        weights_here=[]
        are_there_landmarks=False
        for landmark in landmarks_in_sight:
            landmark_dist, landmark_bearing_angle, landmark_id = landmark
            x,y,theta= self.particles[0].pose
            are_there_landmarks=True
            weights_here=[]
            for particle in self.particles:
                particle.handle_landmark(landmark_dist, landmark_bearing_angle, landmark_id)
                weights_here.append(particle.weight)        
            
        #if are_there_landmarks:
            #print('weights_here: ',weights_here)

        self.particles , self.best_particle_ID = resample(self.particles, self.num_particles, self.resample_method, self.best_particle_ID)
        #use latest estimation to update_screen
        self.update_screen()
        


    def update_screen(self):
        if self.best_particle_ID==-1:
           # x,y,theta= 0,0,0
           self.best_particle_ID=np.random.randint(len(self.particles))

        x,y,theta= self.particles[self.best_particle_ID].pose
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
        #pygame.display.flip()

        #draw current particles
        for particle in self.particles:
            particle_x , particle_y, _ = particle.pose
            pygame.draw.circle(self.screen, self.RED, (int((particle_x ) * self.SCREEN_WIDTH/self.width_meters + self.left_coordinate + self.SCREEN_WIDTH/2), int((particle_y)* self.SCREEN_HEIGHT/self.height_meters+ self.SCREEN_HEIGHT/2)), 3)

        for landmark_id, landmark in self.particles[self.best_particle_ID].landmarks.items():
            landmark_x , landmark_y = landmark.x, landmark.y
            #print('landmark_id',landmark_id, ' pose: ', landmark_x ,' ,', landmark_y)
            pygame.draw.circle(self.screen, self.BLACK, (int(landmark_x* self.SCREEN_WIDTH/self.width_meters + self.left_coordinate + self.SCREEN_WIDTH/2), int(landmark_y* self.SCREEN_HEIGHT/self.height_meters+ self.SCREEN_HEIGHT/2)), 5)
            
            font = pygame.font.Font(None, 30)  
            text_surface = font.render("id:"+str(landmark_id), True, self.BLACK)  # Render text surface
            text_rect = text_surface.get_rect(center=(int(landmark_x * self.SCREEN_WIDTH / self.width_meters + self.left_coordinate + self.SCREEN_WIDTH/2), int(landmark_y * self.SCREEN_HEIGHT / self.height_meters+ self.SCREEN_HEIGHT/2)- 15))  # Position text surface above the circle
            self.screen.blit(text_surface, text_rect)  # Blit text surface onto the screen
            



