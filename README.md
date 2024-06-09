# FastSLAM

## Prerequisites

Before running ArucoSLAM, ensure you have the following software installed:
- ROS (Robot Operating System)
- Python 3
- Required ROS packages:
  - `turtlebot3_teleop`
  - `turtlebot3_gazebo`
  - `cv_bridge`
  - `opencv-python`
  - `numpy`
  - `scipy`
  - `matplotlib`

## How to Run

  **Prepare the Rosbag Directory**

  Place your rosbag files inside this `rosbag` folder, at /Grupo25_SAUT/catkin_ws/src/turtlebot3_fastslam/src/rosbag

  **Run the Program**

  To run the FastSLAM algorithm, use the following command:

  ```sh
  roslaunch turtlebot3_fastslam my_launch.launch rosbag:= <rosbag-file>
  ```
  Replace `<rosbag-file>` with the name of the rosbag file located in the `rosbag` folder. For example:

  ```sh
  roslaunch turtlebot3_fastslam my_launch.launch rosbag:= <Mapa4_2.bag>
  ```

  The script accepts the following arguments:
  - `<rosbag-file>.bag`: Name of the rosbag in the rosbag folder

  In order to observe the output of the FastSLAM algorithm in RVIZ, you should add to RVIZ the following topics:
    /landmarks
    /occupancy_grid
    /raspicam_node/image


## Explanation of Parameters

- **SLAM Variables:**
  - `window_size_pixel`: Pixel size of the window.
  - `size_window`: Size of the window in meters.
  - `OG_map_options`: Occupancy grid map options (width, height, resolution in meters).
  - `number_particles`: Number of particles for the particle filter.
  - `Q_init`: Initial noise covariance matrix.
  - `Q_update`: Update noise covariance matrix.
  - `alphas`: Noise parameters for the motion model.

- **Occupancy Map:**
  - `occupancy_map`: Boolean flag to indicate whether to use occupancy mapping.

## How to Run Micro-Simulation
  **Run the Program**
  Acess the directory: /Grupo25_SAUT/micro_simulation

  To run the microsimulation and FastSLAM algorithm, use the following command:

  ```sh
  python3 main.py
  ```
  To move robot use arrow keys, and space; or WXAD and S keys. To exit, press ESC
