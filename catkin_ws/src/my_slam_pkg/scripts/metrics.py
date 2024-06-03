import numpy as np
from sklearn.metrics import mean_squared_error 
from scipy.spatial.transform import Rotation as R

def horns_method(G, T):
    """"
    G - groundtruth matrix where it has has many rows has points and 2 columns (x,y)
    T -  tranjectory matrix where it has has many rows has points and 2 columns (x,y)
    """

    cg = np.mean(G, axis = 0)
    ct = np.mean(T,axis = 0)

    G_center = G- cg
    T_center = T - ct

    H = (np.dot(T_center.T,G_center)).T
    U,S, Vt = np.linalg.svd(H)

    R_ = U.dot(Vt)
    if np.linalg.det(R_) < 0:
        Vt[-1, :] *= -1
        R_ = U.dot(Vt)
    t_ = cg - R_.dot(ct)

    aligned_t = (R_.dot(T.T)).T + t_

    return aligned_t

def ate(G, T):
    "This only uses x and y coordinates"
    G = G[:,0:2]
    T = T[:,0:2]
    aligned_estimated = horns_method(G,T)
    return mean_squared_error(aligned_estimated,G,squared = False) #False = RMSE

def pose_to_matrix(pose):
    """Convert a (x, y, theta) pose to a 3x3 homogeneous transformation matrix."""
    x, y, theta = pose
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta), np.cos(theta), y],
        [0, 0, 1]
    ])

def rpe(G,T):
    t_errors = []
    r_errors = []
    for i in range(len(G)-1):
        #Fastslam outputs (x,y,theta) but rpe uses Rotation matrixes
        g1 = pose_to_matrix(G[i])
        g2 = pose_to_matrix(G[i+1])
        t1 = pose_to_matrix(T[i])
        t2 = pose_to_matrix(T[i+1])

        delta_t = np.linalg.inv(t1).dot(t2)
        delta_g = np.linalg.inv(g1).dot(g2)

        delta_t_3x3 = np.eye(3)
        delta_t_3x3[:2,:2]=delta_t[:2, :2]

        delta_g_3x3 = np.eye(3)
        delta_g_3x3[:2,:2]=delta_g[:2, :2]

        #Compute the distance value for the inside of the square root
        trans_error = np.linalg.norm(delta_t[:2, 2] - delta_g[:2, 2])
        #Create a rotation object from a Rotation matrix and convert it into a vector
        rot_error = np.linalg.norm(R.from_matrix(delta_t_3x3).as_rotvec() - R.from_matrix(delta_g_3x3).as_rotvec())

        t_errors.append(trans_error)
        r_errors.append(rot_error)
    #Compute root mean squared error of these values
    rpe_trans = np.sqrt(np.mean(np.array(t_errors)**2))
    rpe_rot = np.sqrt(np.mean(np.array(r_errors)**2))
    return (rpe_trans, rpe_rot)



def compute_metrics(slam,ground_truth):
    trajectory = slam.get_trajectory()
    trajectory = np.array(trajectory)
    ate_e = ate(ground_truth, trajectory)
    rpe_e = rpe(ground_truth,trajectory)

    return ate_e, rpe_e

    