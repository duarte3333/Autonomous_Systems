import numpy as np
import math
import random
from scipy import linalg
import copy
import subprocess
import cv2

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(x, y)
    return(rho, np.degrees(phi))
    
def get_rosbag_duration(rosbag_file):
    # Use rosbag info command to get the duration of the rosbag
    result = subprocess.run(['rosbag', 'info', '--yaml', rosbag_file], stdout=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    print("Rosbag info output:", output)  # Debugging line to see the output format
    for line in output.split('\n'):
        if 'duration' in line:
            duration_str = line.split(': ')[1].strip()
            if 'sec' in duration_str:
                duration = float(duration_str.split(' ')[0])
            else:
                parts = duration_str.split(':')
                duration = 0
                if len(parts) == 3:
                    hours, minutes, seconds = map(float, parts)
                    duration = hours * 3600 + minutes * 60 + seconds
                elif len(parts) == 2:
                    minutes, seconds = map(float, parts)
                    duration = minutes * 60 + seconds
                elif len(parts) == 1:
                    duration = float(parts[0])
            return int(duration)
    return 0

def normalize_weights(particles, num_particles):
    sumw = sum([p.weight for p in particles])
    try:
        for i in range(num_particles):
            particles[i].weight /= sumw
    except ZeroDivisionError:
        for i in range(num_particles):
            particles[i].weight = 1.0 / num_particles

        return particles

    return particles

def low_variance_resampling(weights, equal_weights, Num_particles):
    wcum = np.cumsum(weights)
    base = np.cumsum(equal_weights) - 1 / Num_particles
    resampleid = base + np.random.rand(base.shape[0]) / Num_particles
    indices = np.zeros(Num_particles, dtype=int)
    j = 0
    #this will select wich particles to keep.The total nr of particles stays the same. If one is not selected, another should be selected twice
    for i in range(Num_particles):
        while ((j < wcum.shape[0] - 1) and (resampleid[i] > wcum[j])): 
            j+=1                   
        indices[i] = j
    return indices


def stratified_resampling(weights, Num_particles):
    """
    Stratified resampling algorithm
    """
    cumulative_weights = np.cumsum(weights)
    strata = np.linspace(0, 1, Num_particles + 1)[:-1] + np.random.rand(Num_particles) / Num_particles
    indices = np.zeros(Num_particles, dtype=int)
    j = 0
    for i in range(Num_particles):
        while cumulative_weights[j] < strata[i]:
            j += 1
        indices[i] = j
    return indices




def resample(particles, Num_particles, resample_method,new_highest_weight_index):
    #for p in particles:
     #   p.weight=np.random.normal(0.5,0.5,1)
    particles = normalize_weights(particles, Num_particles)
    weights = np.array([particle.weight for particle in particles])
    highest_weight_index = np.argmax(weights)  # Index of particle with highest weight before equalization


    # Neff = 1.0 / (weights @ weights.T)  # Effective particle number
    # equal_weights=np.array(weights*0 + 1/Num_particles)
    # Neff_maximum = 1.0 / (equal_weights @ equal_weights.T)
    Neff = 1.0 / np.sum(np.square(weights))  # Effective particle number
    equal_weights = np.full_like(weights, 1 / Num_particles)
    Neff_maximum = 1.0 / np.sum(np.square(equal_weights))

    #print('Neff',Neff,'   ', Neff_maximum)
    if Neff < Neff_maximum/2:  # only resample if Neff is too low - partiles are not represantative os posteriori
        #print('Lets resample')
        if resample_method=="low variance":
            indices = low_variance_resampling(weights, equal_weights, Num_particles)
        elif resample_method=="Stratified":
            indices=stratified_resampling(weights, Num_particles)
        
        particles_copy = copy.deepcopy(particles)
        for i in range(len(indices)):
            particles[i].pose = particles_copy[indices[i]].pose
            particles[i].landmarks = particles_copy[indices[i]].landmarks
            particles[i].weight = 1.0 / Num_particles   #only time that weight is reset is here
            if highest_weight_index == indices[i]:
                new_highest_weight_index=i
    return particles , new_highest_weight_index

def gauss_noise(mu, sig):
    """ This function generates a random number from a gaussian 
    distribution with mean mu and standard deviation sig"""
    return random.gauss(mu, sig)

def euclidean_distance(a, b):
    """ This function calculates the euclidean distance between two points"""
    return math.hypot(b[0]-a[0], b[1]-a[1])

def cal_direction(a, b):
    """Calculate the angle of the vector a to b"""
    return math.atan2(b[1]-a[1], b[0]-a[0])

def multi_normal(x, mean, cov):
    """Calculate the density for a multinormal distribution"""
    den = 2 * math.pi * math.sqrt(linalg.det(cov))
    num = np.exp(-0.5*np.transpose((x - mean)).dot(linalg.inv(cov)).dot(x - mean))
    result = num/den
    return result[0][0]

def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
