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

    Create a folder named `rosbag` inside the `my_slam_pkg` folder. Place your rosbag files inside this `rosbag` folder.

    **Run the Program**

    To run ArucoSLAM, use the following command:

    ```sh
    python3 main.py <rosbag-file>
    ```

    Replace `<rosbag-file>` with the name of the rosbag file located in the `rosbag` folder. For example:

    ```sh
    python3 main.py example.bag
    ```

    The script accepts the following arguments:
    - `<rosbag-file>.bag`: Name of the rosbag in the rosbag folder
    - `live`: For live teleoperation.
    - `microsim`: For running microsimulation.

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

## Example Command

To run the SLAM process with a specific rosbag file, use:

```sh
python3 aruco_slam.py example.bag
```