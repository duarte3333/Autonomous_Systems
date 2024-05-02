import numpy as np

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
