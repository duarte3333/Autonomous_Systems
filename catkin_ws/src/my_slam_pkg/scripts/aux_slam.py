import numpy as np
import math
import random
from scipy import linalg
import copy
import subprocess
import cv2

# Convert Cartesian coordinates to polar coordinates
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)  # Compute the radius
    phi = np.arctan2(x, y)  # Compute the angle
    return(rho, np.degrees(phi))  # Return radius and angle in degrees
    
# Get the duration of a rosbag file using the rosbag info command
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

# Normalize the weights of the particles
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

# Low variance resampling method
def low_variance_resampling(weights, equal_weights, Num_particles):
    wcum = np.cumsum(weights)
    base = np.cumsum(equal_weights) - 1 / Num_particles
    resampleid = base + np.random.rand(base.shape[0]) / Num_particles
    indices = np.zeros(Num_particles, dtype=int)
    j = 0
    # This will select which particles to keep. The total number of particles stays the same. 
    # If one is not selected, another should be selected twice.
    for i in range(Num_particles):
        while ((j < wcum.shape[0] - 1) and (resampleid[i] > wcum[j])): 
            j += 1                   
        indices[i] = j
    return indices

# Stratified resampling method
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

# Resampling function
def resample(particles, Num_particles, resample_method, new_highest_weight_index):
    # Normalize weights
    particles = normalize_weights(particles, Num_particles)
    weights = np.array([particle.weight for particle in particles])
    highest_weight_index = np.argmax(weights)  # Index of particle with highest weight before equalization

    # Calculate effective particle number
    Neff = 1.0 / np.sum(np.square(weights))  # Effective particle number
    equal_weights = np.full_like(weights, 1 / Num_particles)
    Neff_maximum = 1.0 / np.sum(np.square(equal_weights))

    # Resample if Neff is too low - particles are not representative of posterior
    if Neff < Neff_maximum / 2:
        if resample_method == "low variance":
            indices = low_variance_resampling(weights, equal_weights, Num_particles)
        elif resample_method == "Stratified":
            indices = stratified_resampling(weights, Num_particles)
        
        particles_copy = copy.deepcopy(particles)
        for i in range(len(indices)):
            particles[i].pose = particles_copy[indices[i]].pose
            particles[i].landmarks = particles_copy[indices[i]].landmarks
            particles[i].weight = 1.0 / Num_particles  # Only time that weight is reset is here
            if highest_weight_index == indices[i]:
                new_highest_weight_index = i
    return particles, new_highest_weight_index

# Generate Gaussian noise
def gauss_noise(mu, sig):
    """ This function generates a random number from a Gaussian 
    distribution with mean mu and standard deviation sig"""
    return random.gauss(mu, sig)

# Calculate Euclidean distance between two points
def euclidean_distance(a, b):
    """ This function calculates the Euclidean distance between two points"""
    return math.hypot(b[0] - a[0], b[1] - a[1])

# Calculate the direction angle from point a to point b
def cal_direction(a, b):
    """Calculate the angle of the vector a to b"""
    return math.atan2(b[1] - a[1], b[0] - a[0])

# Calculate the density for a multivariate normal distribution
def multi_normal(x, mean, cov):
    """Calculate the density for a multivariate normal distribution"""
    den = 2 * math.pi * math.sqrt(linalg.det(cov))
    num = np.exp(-0.5 * np.transpose((x - mean)).dot(linalg.inv(cov)).dot(x - mean))
    result = num / den
    return result[0][0]

# Normalize an angle to the range [-pi, pi]
def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle
