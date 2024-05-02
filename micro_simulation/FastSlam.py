import pygame
import math 

class FastSlam:
    def __init__(self, window_size_pixel, sample_rate, size_m,central_bar_width, screen ):
        
        # Define colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 160, 0)
        self.BLUE=(10,10,255)

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
        self.turtlebot_radius_pixel =self.turtlebot_radius * self.SCREEN_WIDTH/self.width_meters #from turtlebot website

        self.update_screen(0,0,0)





        #initialize FastSlam variables



        return
    def compute(self, odometry, landmarks_in_sight):
        #compute and display FastSlam



        #use latest estimation to update_screen
        #update_screen(x_predicted,y_predicted,theta_predicted)
        self.update_screen(0,0,0)
        return


    def update_screen(self, x, y, theta):
        
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
    