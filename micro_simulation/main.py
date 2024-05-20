import numpy as np
import pygame
from FastSlam import FastSlam
from Simulation import Simulation
from aux_1 import *

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
def init_variables():
    
    #Main parameters:
    Odometry_noise= True  #Gausian noise na odometria
    landmark_noise=True
    only_slam_window=True
    window_size_pixel=1000    #tamanho da janela
    sample_rate=100  #sample rate (Hz)
    size_m = 3#float(input('What should be the size of the map? n x n (in meters). n is: '))
    nr_landmarks =5# int(input('How many random arucu landmarks do you want in your map?'))
    central_bar_width=10
    number_particles=30

    landmarks = create_landmarks(nr_landmarks,size_m,size_m)
    sim=Simulation(only_slam_window,size_m, size_m,window_size_pixel, Odometry_noise,landmark_noise, sample_rate, central_bar_width)
    my_slam = FastSlam(only_slam_window,window_size_pixel, sample_rate, size_m, central_bar_width,sim.turtlebot.wheel_base , number_particles, sim.get_screen())
    return landmarks,sim, my_slam

def run_simulation_main():
    pygame.init()
    clock = pygame.time.Clock()
    landmarks,sim, my_slam=init_variables()
    count=0
    while sim.get_running():
        sim.loop_iteration(landmarks)
        if count<10:
            my_slam.update_odometry(sim.get_odometry())
            count+=1
        else: #update landmarks with less frequency than the odometry
            my_slam.compute_slam(sim.get_odometry(),sim.get_Landmarks_in_sight(landmarks, "Relative_pose") )
            count=0
        clock.tick(1/sim.time_interval)

    pygame.display.quit()
    pygame.quit()
    
if __name__ == "__main__":
    run_simulation_main()
 