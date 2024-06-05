import numpy as np
from aux_slam import *

def is_too_close(point, landmarks, min_distance):
    for landmark in landmarks:
        if euclidean_distance(point[0], landmark) < min_distance:
            return True
    return False

def create_landmarks(nr_landmarks, width, height):
    min_distance = 0.2
    landmarks = []
    while len(landmarks) < nr_landmarks:
        x_y_proposal = np.random.rand(1, 2) #this returns a random number between 0 and 1
        x_y_proposal *= np.array([width, height]) 
        if not is_too_close(x_y_proposal, landmarks, min_distance):
            landmarks.append(tuple(x_y_proposal[0]))
    return landmarks

