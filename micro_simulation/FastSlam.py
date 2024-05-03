import pygame
import math 
import numpy as np
from Particule import Particle

class FastSlam:
    def __init__(self, window_size_pixel, sample_rate, size_m,central_bar_width, turtlebot_L, screen,std_dev_motion = 0.2, num_particles=100 ):
        self.std_dev_motion = std_dev_motion

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

        self.update_screen(0,0,0)

        self.num_particles = num_particles
        self.particles = self.initialize_particles()
        self.best_particle_ID=-1


        #initialize FastSlam variables



        return
    
    def initialize_particles(self, landmarks={}):
        particles = []
        for _ in range(self.num_particles):
            # Initialize each particle with a random pose and empty landmarks
            x = np.random.uniform(0, self.width_meters)
            y = np.random.uniform(0, self.height_meters)
            theta = np.random.uniform(0, 2 * np.pi)
            pose = np.array([x, y, theta])
            particles.append(Particle(pose, landmarks, self.turtlebot_L,self.std_dev_motion ))
        return particles
    
   
    def update_odometry(self,odometry):
        # Update each particle with motion and observation models
        for particle in self.particles:
            # Motion update
            particle.motion_model(odometry)
            
        self.update_screen()


        
    def resample(self, landmarks_in_sight):
        weight=[]
        for particle in self.particles:
            weight.append(particle.compute_weight(landmarks_in_sight))
        
        #define self.best_particle_ID
        #use weight list to resample particles
        #...






    def compute(self, odometry, landmarks_in_sight):
        #compute and display FastSlam
        self.update_odometry(odometry)
        # Landmark update
        for landmark in landmarks_in_sight:
            landmark_position_x, landmark_position_y, landmark_id = landmark
            for particle in self.particles:
                particle.update_landmark(landmark_position_x, landmark_position_y, landmark_id)
        
        self.resample(landmarks_in_sight)
        #use latest estimation to update_screen
        #update_screen(x_predicted,y_predicted,theta_predicted)
        self.update_screen()
        


    def update_screen(self):
        x,y,theta= self.particles[self.best_particle_ID].pose
        # Calculate the vertices of the triangle for orientation
        turtlebot_pos= (int((x +  self.width_meters/2) * self.SCREEN_WIDTH/self.width_meters +self.left_coordinate), int((y +self.height_meters/2) *self.SCREEN_HEIGHT/self.height_meters)) #window should display a 5x5 m^2 area
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
        pygame.display.flip()

        #draw current particles
        for particle in self.particles:
            particle_x , particle_y, _ = particle.pose
            pygame.draw.circle(self.screen, self.RED, (int((particle_x +  self.width_meters/2) * self.SCREEN_WIDTH/self.width_meters + self.left_coordinate), int((particle_y+self.height_meters/2)* self.SCREEN_HEIGHT/self.height_meters)), 1)
        
    