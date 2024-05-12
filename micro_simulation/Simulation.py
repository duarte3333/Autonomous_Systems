import pygame
import math 
import numpy as np
from Turtlebot_model import TurtleBot3Waffle
from aux_1 import *
from aux_slam import euclidean_distance


class Simulation:
        
    def __init__(self, width_meters=5, height_meters=5, size_pixel=800, Odometry_noise=True, Landmark_noise=True, sample_rate=60, central_bar_width=10):
        pygame.init()

        # Define colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 160, 0)
        self.BLUE=(10,10,255)

        # Set up the screen
        self.SCREEN_WIDTH = size_pixel
        self.SCREEN_HEIGHT = size_pixel
        self.width_meters=width_meters 
        self.height_meters=height_meters
        self.turtlebot_radius=0.105 
        self.turtlebot_radius_pixel =self.turtlebot_radius * self.SCREEN_WIDTH/width_meters #from turtlebot website

        pygame.display.init()
        self.central_bar_width=central_bar_width
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH*2 + self.central_bar_width, self.SCREEN_HEIGHT))
        pygame.display.set_caption("TurtleBot3 Micro-Simulation and Slam predictions")
        self.indices_in_sight=[]

        self.linear_velocity = 0
        self.angular_velocity = 0

        self.time_interval = 1.0 / sample_rate  # sample rate in Hz
        self.turtlebot = TurtleBot3Waffle(self.time_interval,Odometry_noise,width_meters, height_meters, self.turtlebot_radius )
        self.running=True
        self.clock = pygame.time.Clock()
        self.Landmark_noise=Landmark_noise
        self.std_dev_landmark = 0.05

    def loop_iteration(self, landmarks):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.linear_velocity += 0.02
                elif event.key == pygame.K_x or event.key == pygame.K_DOWN:
                    self.linear_velocity += -0.02

                elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                    self.angular_velocity += math.pi / 20
                elif event.key == pygame.K_d  or event.key == pygame.K_RIGHT:
                    self.angular_velocity += -math.pi / 20
                elif event.key ==pygame.K_s  or event.key == pygame.K_SPACE:
                    self.angular_velocity=0
                    self.linear_velocity = 0
        
        self.turtlebot.move(self.linear_velocity,self.angular_velocity, False)
        self.indices_in_sight = self.turtlebot.check_landmarks(landmarks)
        #print(self.indices_in_sight)
        self.update_screen(landmarks)


    def update_screen(self, landmarks):
        x,y = self.turtlebot.get_position()
        theta = self.turtlebot.get_orientation()

        # Calculate the vertices of the triangle for orientation
        turtlebot_pos= (int((x +  self.width_meters/2) * self.SCREEN_WIDTH/self.width_meters), int((y +self.height_meters/2) *self.SCREEN_HEIGHT/self.height_meters)) #window should display a 5x5 m^2 area
        
        triangle_length = 0.8*self.turtlebot_radius_pixel
        triangle_tip_x = turtlebot_pos[0] + triangle_length * math.cos(theta) 
        triangle_tip_y = turtlebot_pos[1] - triangle_length * math.sin(theta)
        triangle_left_x = turtlebot_pos[0] + triangle_length * math.cos(theta + 5 * math.pi / 6) 
        triangle_left_y = turtlebot_pos[1] - triangle_length * math.sin(theta + 5 * math.pi / 6) 
        triangle_right_x = turtlebot_pos[0] + triangle_length * math.cos(theta - 5 * math.pi / 6)
        triangle_right_y = turtlebot_pos[1] - triangle_length * math.sin(theta - 5 * math.pi / 6)

        # Draw the triangle
        triangle_points = [(triangle_tip_x, triangle_tip_y), (triangle_left_x, triangle_left_y), (triangle_right_x, triangle_right_y)]

        #self.screen.fill(self.WHITE)
          # Fill only half of the screen with a color
        half_screen_rect = pygame.Rect(0, 0, self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.WHITE, half_screen_rect)

        bar_rect = pygame.Rect(self.SCREEN_WIDTH , 0, self.central_bar_width, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.BLACK, bar_rect)
        pygame.draw.circle(self.screen, self.GREEN, turtlebot_pos , self.turtlebot_radius_pixel)
        pygame.draw.polygon(self.screen, self.BLUE, triangle_points)
        id=0
        for landmark in landmarks:
            landmark_x , landmark_y = landmark
            pygame.draw.circle(self.screen, self.BLACK, (int(landmark_x* self.SCREEN_WIDTH/self.width_meters), int(landmark_y* self.SCREEN_HEIGHT/self.height_meters)), 5)
            # Render id text
            font = pygame.font.Font(None, 30)  # You can change the font and size here
            text_surface = font.render("id:"+str(id), True, self.BLACK)  # Render text surface
            text_rect = text_surface.get_rect(center=(int(landmark_x * self.SCREEN_WIDTH / self.width_meters), int(landmark_y * self.SCREEN_HEIGHT / self.height_meters) - 15))  # Position text surface above the circle
            self.screen.blit(text_surface, text_rect)  # Blit text surface onto the screen
            
            id+=1
        for i in self.indices_in_sight:
            landmark_x , landmark_y = landmarks[i]
            pygame.draw.circle(self.screen, self.BLACK, (int(landmark_x* self.SCREEN_WIDTH/self.width_meters), int(landmark_y* self.SCREEN_HEIGHT/self.height_meters)), 10)
        

        pygame.display.flip()        # Update the display
        self.clock.tick(1/self.time_interval)

    def get_running(self):
        return self.running
    def get_odometry(self):
        return self.turtlebot.get_odometry()
    def get_screen(self):
        return self.screen
    
    def get_Landmarks_in_sight(self, landmarks, abs_or_relative):
        Landmarks_in_sight=[]
        #print("indices_in_sight", self.indices_in_sight)
        if abs_or_relative=="Absolute_pose":
            for indice in self.indices_in_sight:
                Landmarks_in_sight.append([landmarks[indice][0], landmarks[indice][1], indice])
        elif abs_or_relative=="Relative_pose":
            my_x, my_y = self.turtlebot.get_position()
            my_x, my_y =  my_x + self.width_meters/2, my_y + self.height_meters/2
            my_theta =self.turtlebot.get_orientation()

            for indice in self.indices_in_sight:
                landmark_x=landmarks[indice][0]
                landmark_y=landmarks[indice][1]
                landmark_ID=indice

                angle_to_landmark = -math.atan2(landmark_y - my_y, landmark_x - my_x)
                beta = angle_to_landmark - my_theta

                #print(my_theta, " ", angle_to_landmark)
                distance_to_landmark = euclidean_distance([my_x, my_y],landmarks[indice])
                #print("dist ",distance_to_landmark)

                #x_rel = distance_to_landmark * math.cos(angle_difference)
                #y_rel = distance_to_landmark * math.sin(angle_difference)    
                if  self.Landmark_noise:
                    noise_angle =  np.random.normal(0, self.std_dev_landmark, 1)*beta
                    noise_dist= np.random.normal(0, self.std_dev_landmark, 1)*distance_to_landmark
                else:
                    noise_angle=[0]
                    noise_dist=[0]
                
                Landmarks_in_sight.append((distance_to_landmark+ noise_dist[0] ,beta + noise_angle[0], landmark_ID))
                #print('Landmarks_in_sight', Landmarks_in_sight)

        return np.array(Landmarks_in_sight)