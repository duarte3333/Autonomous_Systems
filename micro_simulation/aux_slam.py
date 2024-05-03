import numpy as np
import math
import random
from scipy import linalg


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