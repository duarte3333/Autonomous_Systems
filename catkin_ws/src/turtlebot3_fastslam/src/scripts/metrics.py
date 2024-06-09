import numpy as np
from sklearn.metrics import mean_squared_error 
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

# Horn's method to align two sets of points using SVD
def horns_method(G, T):
    """
    G - groundtruth matrix with rows as points and 2 columns (x, y)
    T - trajectory matrix with rows as points and 2 columns (x, y)
    """
    cg = np.mean(G, axis=0)  # Centroid of ground truth
    ct = np.mean(T, axis=0)  # Centroid of trajectory

    G_center = G - cg  # Centered ground truth
    T_center = T - ct  # Centered trajectory

    H = (np.dot(T_center.T, G_center)).T
    U, S, Vt = np.linalg.svd(H)

    R_ = U.dot(Vt)
    if np.linalg.det(R_) < 0:
        Vt[-1, :] *= -1
        R_ = U.dot(Vt)
    t_ = cg - R_.dot(ct)

    aligned_t = (R_.dot(T.T)).T + t_

    return aligned_t

# Compute Absolute Trajectory Error (ATE)
def ate(G, T):
    "This only uses x and y coordinates"
    G = G[:, 0:2]
    T = T[:, 0:2]
    aligned_estimated = horns_method(G, T)
    return mean_squared_error(aligned_estimated, G, squared=False)  # RMSE

# Convert pose to a 3x3 homogeneous transformation matrix
def pose_to_matrix(pose):
    x, y, theta = pose
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])

# Compute Relative Pose Error (RPE)
def rpe(G, T):
    t_errors = []
    r_errors = []
    for i in range(len(G) - 1):
        # Convert poses to transformation matrices
        g1 = pose_to_matrix(G[i])
        g2 = pose_to_matrix(G[i + 1])
        t1 = pose_to_matrix(T[i])
        t2 = pose_to_matrix(T[i + 1])

        delta_t = np.linalg.inv(t1).dot(t2)
        delta_g = np.linalg.inv(g1).dot(g2)

        delta_t_3x3 = np.eye(3)
        delta_t_3x3[:2, :2] = delta_t[:2, :2]

        delta_g_3x3 = np.eye(3)
        delta_g_3x3[:2, :2] = delta_g[:2, :2]

        # Compute translation error
        trans_error = np.linalg.norm(delta_t[:2, 2] - delta_g[:2, 2])
        # Compute rotation error
        rot_error = np.linalg.norm(R.from_matrix(delta_t_3x3).as_rotvec() - R.from_matrix(delta_g_3x3).as_rotvec())

        t_errors.append(trans_error)
        r_errors.append(rot_error)
    
    # Compute RMSE for translation and rotation errors
    rpe_trans = np.sqrt(np.mean(np.array(t_errors)**2))
    rpe_rot = np.sqrt(np.mean(np.array(r_errors)**2))
    return (rpe_trans, rpe_rot)

# Ground truth landmarks for different maps
def landmarks_PoseMeasured(map_nr):
    if map_nr == 1:
        x = np.array([1.06, -0.88, -0.5, -0.47, 0])
        y = np.array([0.61, 0.5, -0.32, 1.93, 0])
    elif map_nr == 2:
        x = np.array([-0.9, -0.6, -1.15, -0.15, 0])
        y = np.array([3.6, -1.3, -0.66, 3.9, 0])
    elif map_nr == 3:
        x = np.array([0, 2.27, 1, 3.29, 3.8])
        y = np.array([0, 0.82, 1.96, -0.74, 0])
    elif map_nr == 4:
        x = np.array([0.65, 0.64, 2.74, 0, 2.74])
        y = np.array([5.5, -0.05, 1.41, 0.78, 0.31])
    else:
        print('Not a valid map nr (must be either 1, 2, 3, 4). SSE is not going to be valid')
        x = np.array([0, 0, 0, 0, 0])
        y = np.array([0, 0, 0, 0, 0])
    IDS = np.array([15, 53, 60, 77, 100])
    return np.stack((IDS, x, y))

# Perform Singular Value Decomposition (SVD) to find the rigid transformation
def SVD_rigidTransform(A, B):
    centroid_A = np.mean(A, axis=1).reshape(2, 1)
    centroid_B = np.mean(B, axis=1).reshape(2, 1)
    
    A_prime = A - centroid_A
    B_prime = B - centroid_B
   
    H = A_prime @ B_prime.T
    U, _, Vt = np.linalg.svd(H)
    V = Vt.T
    
    Rot = V @ U.T
    t = centroid_B - Rot @ centroid_A

    return Rot, t

# Apply rotation and translation to a set of points
def applyRT(matrix, Rot, T):
    rotated_matrix = Rot @ matrix
    return rotated_matrix + T

# Compute Sum of Squared Errors (SSE) between two sets of points
def SSE(A, B):
    SSE_metric = 0
    count = 0
    for i in range(A.shape[1]):
        SSE_metric += (A[0, i] - B[0, i])**2 + (A[1, i] - B[1, i])**2
        count += 1
    return SSE_metric

# Remove columns from matrix A that are not in matrix B
def remove_columns(A_to_remove, B):
    to_remove = []
    for i in range(A_to_remove.shape[1]):
        flag = False
        for j in range(B.shape[1]):
            if B[0, j] == A_to_remove[0, i]:
                flag = True
        if flag == False:
            to_remove.append(i)
    A_new = np.delete(A_to_remove, to_remove, 1)
    return A_new

# Ensure both matrices have the same number of columns by removing extra columns
def check_arraySize(A, B):
    if A.shape[1] > B.shape[1]:
        A_new = remove_columns(A, B)
        B_new = B
    elif A.shape[1] < B.shape[1]:
        B_new = remove_columns(B, A)
        A_new = A
    else:
        A_new = A
        B_new = B
    return A_new, B_new

# Compute metrics for landmarks
def landmark_metrics(map_nr, slam):
    slam_landmarks = slam.my_slam.get_BestLandmarks()
    landmarks_GroundTruth = landmarks_PoseMeasured(map_nr)
    slam_landmarks, landmarks_GroundTruth = check_arraySize(slam_landmarks, landmarks_GroundTruth)
   
    Rot, T = SVD_rigidTransform(slam_landmarks[1:3, :], landmarks_GroundTruth[1:3, :])
    slam_landmarks = applyRT(slam_landmarks[1:3, :], Rot, T)

    SSE_metric = SSE(slam_landmarks, landmarks_GroundTruth[1:3, :])
    return SSE_metric

# Parse ground truth data from a text file
def parsing_txt(filename):
    filepath = 'groundtruth/' + filename
    with open(filepath, 'r') as file:
        lines = file.readlines()

        # Convert each line to a tuple
        tuple_list = []
        for line in lines:
            line = line.strip()
            tuple_element = eval(line)
            tuple_list.append(tuple_element)

        # Convert the list of tuples to a NumPy array
        groundtruth = np.array(tuple_list)
    
    return groundtruth

# Drop rows from the larger array to match the size of the smaller array
def drop_rows(groundtruth, trajectory):
    number_rows = len(groundtruth) - len(trajectory)
    flag = 0
    if number_rows > 0:  # groundtruth has more points
        num_total_rows = groundtruth.shape[0]
        random_indices = np.random.choice(num_total_rows, number_rows, replace=False)
        groundtruth = np.delete(groundtruth, random_indices, axis=0)
    elif number_rows < 0:  # trajectory has more points
        num_total_rows = trajectory.shape[0]
        random_indices = np.random.choice(num_total_rows, number_rows, replace=False)
        trajectory = np.delete(trajectory, random_indices, axis=0)
    return groundtruth, trajectory

# Compute trajectory metrics (ATE and RPE)
def trajectory_metrics(file, slam):
    txt_file = file.replace('.bag', '.txt')
    groundtruth = parsing_txt(txt_file)
    trajectory = slam.get_trajectory()
    trajectory = np.array(trajectory)

    groundtruth, trajectory = drop_rows(groundtruth, trajectory)

    ate_e = ate(groundtruth, trajectory)
    rpe_e = rpe(groundtruth, trajectory)

    return ate_e, rpe_e

# Compute manual distance error
def manual_distance(slam):
    trajectory = slam.get_trajectory()
    trajectory = np.array(trajectory)
    first_point = trajectory[0, :2]
    last_point = trajectory[-1, :2]
    distance = np.linalg.norm(first_point - last_point)
    error = abs(distance - 4.04)
    return error

# Compute all relevant metrics for the SLAM
def compute_metrics(slam, map_nr, file):
    if map_nr == 5:
        error = manual_distance(slam)
        return error
    ate_e, rpe_e = trajectory_metrics(file, slam)
    SSE_landmarks = landmark_metrics(map_nr, slam)
    
    return ate_e, rpe_e, SSE_landmarks
