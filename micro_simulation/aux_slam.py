import numpy as np
import math
import random
from scipy import linalg
from Particule import Particle

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
    weights = []
    
    for i in range(Num_particles):
        weights.append(particles[i].weight)
    weights = np.array(weights).T

    highest_weight_index = np.argmax(weights)  # Index of particle with highest weight before equalization


    Neff = 1.0 / (weights @ weights.T)  # Effective particle number
    equal_weights=np.array(weights*0 + 1/Num_particles)
    Neff_maximum = 1.0 / (equal_weights @ equal_weights.T)
    #print('Neff',Neff,'   ', Neff_maximum)
    if Neff < Neff_maximum/2:  # only resample if Neff is too low - partiles are not represantative os posteriori
        if resample_method=="low variance":
            indices = low_variance_resampling(weights, equal_weights, Num_particles)
        elif resample_method=="Stratified":
            indices=stratified_resampling(weights, Num_particles)
        
        particles_copy = particles[:]
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