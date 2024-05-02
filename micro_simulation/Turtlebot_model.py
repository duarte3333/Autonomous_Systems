import math
import numpy as np
import pygame
from FastSlam import FastSlam

#Instruções para usar este microsimulador:
""" 
-Odometry retorna valores em metros que a roda da esquerda e da direita andaram até agora. Isto tem incluido ruido gausiano
O ruido gausiano pode ser desligado com uma variável na função main

Deve-se definir o sample rate também na função main

sim.get_Landmarks_in_sight(landmarks, "Relative_pose") serve para obter um numpy array com as coordenadas dos landmarks que estamos a ver.
-Ao pedir "Relative_pose", as coordenadas dadas serão relativas à posição e orientação da camera do turtlebot
-Ao pedir "Absolute_pose", as coordenadas dadas para as landmarks serão absolutas no mapa da simulação
A opção "Relative_pose" é a que se aproxima melhor da situação do turtlebot real, já que este diz apenas onde está a landmark em relação a si proprio.

 """
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






class Simulation:
        
    def __init__(self, width_meters=5, height_meters=5, size_pixel=800, Odometry_noise=True, sample_rate=60, central_bar_width=10):
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

    def loop_iteration(self, landmarks):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if event.key == pygame.K_w or event.key == pygame.K_UP:
                    self.linear_velocity += 0.01
                elif event.key == pygame.K_x or event.key == pygame.K_DOWN:
                    self.linear_velocity += -0.01

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

        for landmark in landmarks:
            landmark_x , landmark_y = landmark
            pygame.draw.circle(self.screen, self.BLACK, (int(landmark_x* self.SCREEN_WIDTH/self.width_meters), int(landmark_y* self.SCREEN_HEIGHT/self.height_meters)), 5)
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
                Landmarks_in_sight.append(landmarks[indice])
        elif abs_or_relative=="Relative_pose":
            my_x, my_y = self.turtlebot.get_position()
            my_x, my_y =  my_x + self.width_meters/2, my_y + self.height_meters/2
            my_theta =self.turtlebot.get_orientation()

            for indice in self.indices_in_sight:
                landmark_x=landmarks[indice][0]
                landmark_y=landmarks[indice][1]
                angle_to_landmark = -math.atan2(landmark_y - my_y, landmark_x - my_x)
                angle_difference = my_theta - angle_to_landmark
                #print(my_theta, " ", angle_to_landmark)
                distance_to_landmark = distance([my_x, my_y],landmarks[indice])
                #print("dist ",distance_to_landmark)

                x_rel = distance_to_landmark * math.cos(angle_difference)
                y_rel = distance_to_landmark * math.sin(angle_difference)                
                Landmarks_in_sight.append((x_rel,y_rel))

        return np.array(Landmarks_in_sight)
    

def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0] )**2 + (point1[1] - point2[1])**2)

def is_too_close(point, landmarks, min_distance):
    for landmark in landmarks:
        if distance(point[0], landmark) < min_distance:
            return True
    return False

def create_landmarks(nr_landmarks, width, height):
    min_distance = 0.2
    landmarks = []
    while len(landmarks) < nr_landmarks:
        x_y_proposal = np.random.rand(1, 2)
        x_y_proposal *= np.array([width, height])
        if not is_too_close(x_y_proposal, landmarks, min_distance):
            landmarks.append(tuple(x_y_proposal[0]))
    return landmarks


def run_simulation_main():
    #Main parameters:
    Odometry_noise= True    #Gausian noise na odometria
    window_size_pixel=700    #tamanho da janela
    sample_rate=100    #sample rate (Hz)
    size_m = float(input('What should be the size of the map? n x n (in meters). n is: '))
    nr_landmarks = int(input('How many random arucu landmarks do you want in your map?'))
    central_bar_width=10

    landmarks = create_landmarks(nr_landmarks,size_m,size_m)
    sim=Simulation(size_m, size_m,window_size_pixel, Odometry_noise, sample_rate, central_bar_width)
    my_slam = FastSlam(window_size_pixel, sample_rate, size_m, central_bar_width, sim.get_screen())

    while sim.get_running():
        sim.loop_iteration(landmarks)
        my_slam.compute(sim.get_odometry(),sim.get_Landmarks_in_sight(landmarks, "Relative_pose") )
        #print('Odometry:', sim.get_odometry() )  THIS IS USED TO GET ODOMETRY
        #print('Landmarks in sight ', sim.get_Landmarks_in_sight(landmarks, "Relative_pose")) #THIS IS USED TO GET LANDMARKS POSITION

    pygame.display.quit()
    pygame.quit()


if __name__ == "__main__":
    run_simulation_main()
 
    