# Graph-Based SLAM for Autonomous Drone Racing

This repository contains the source code and documentation for the senior thesis "Graph-Based SLAM for Autonomous Drone Racing" by Robert Azarcon from the University of Illinois at Urbana-Champaign. This project proposes and evaluates a Simultaneous Localization and Mapping (SLAM) system designed for autonomous drone racing, utilizing a factor graph optimization framework. Currently, it only runs in offline mode.

Read the thesis [here](Undergrad_Senior_Thesis.pdf)

## About The Project

The goal of this research is to enable a drone equipped with only a camera and an IMU sensor to localize itself and map the positions of gates on a race course. The system was developed and tested in a custom simulation environment using Unreal Engine 5.

### Key Features:
* **Gate Detection**: A Convolutional Neural Network (Faster R-CNN) is used to detect the corner points of race gates from the drone's camera feed.
* **Visual-Inertial Odometry (VIO)**: A tightly coupled VIO module (VINS-FUSION) runs in parallel to generate odometry data.
* **Factor Graph Optimization**: Gate corner detections and odometry are integrated over time in a factor graph to simultaneously estimate the drone's trajectory and the 3D positions of the gates.

## System Design

The SLAM system is composed of a "Front End" and a "Back End".

### Front End
The Front End processes visual data to identify and track features on the race gates.

1. **Gate Corner Detection**: Faster R-CNN, trained on the 2019 AlphaPilot Challenge dataset, detects gate corners.
2. **Corner Point Refinement**: A refinement step uses the Hough transform to find the precise intersection of the inner edges of the gate corners within the detected bounding boxes.
3. **Gate Registration and Tracking**: The 3D points of the corners are geometrically verified to match the known dimensions of the gates. The verified corners are then tracked across frames using ByteTrack.

### Back End
The Back End constructs and solves a factor graph to estimate the drone's pose and gate locations.

* **Visual Projection Factors**: These factors constrain the estimated 3D corner positions to the current drone pose.
* **Ranging Factors**: The known physical dimensions of the gates are used to add rigid constraints between the estimated corner positions.
* **Between Factors**: Odometry information from VINS-FUSION is used to constrain temporally adjacent drone poses.

## Implementation

The system is implemented in Python 3.10 and relies on the following software packages:

| Software | Version | Purpose |
| :--- | :--- | :--- |
| NumPy | 1.26.4 | Array algorithms and frame transformations |
| GTSAM | 4.2 | Factor graph modeling and optimization |
| OpenCV | 4.11.0 | Image filtering, transformations, Hough transform, and non-maximum suppression |
| PyTorch | 2.6.0 | FasterRCNN training and inference |
| ByteTrack | 1.0 | Bounding box tracking |
| VINS-FUSION | 1.0 | Visual Inertial Odometry (VIO) |
| Cosys-AirSim | 3.2.0 | Drone Simulation in Unreal Engine 5 |

### Simulation Environment

The experiments were conducted in a modified version of the Unreal Engine 5's blocks world using the Cosys-AirSim simulator. The environment was set up with three drone gates placed at varying distances. The drone flew a straight-line trajectory towards the third gate.

## Results

The proposed SLAM system successfully estimated the drone's position and the location of the gates.

* The inclusion of gates as SLAM landmarks significantly reduced the positional drift compared to the VIO-only solution.
* The average positional error of the drone was reduced from 0.65 meters (VINS-FUSION only) to 0.27 meters with the proposed SLAM system. This is supported by the root mean square error (RMSE) of the L2 norm, which was 0.6576m for VIO and 0.2764m for the proposed SLAM.
* The estimated positions of the gates were determined with an average error of approximately 0.42 meters.
* There was a slight increase in the average angular error of about 0.35 degrees when using the proposed SLAM method.

## Citation

If you use this work, please cite the original thesis:

Azarcon, R. (2025). *Graph-Based SLAM for Autonomous Drone Racing* (Bachelor's thesis). University of Illinois at Urbana-Champaign.
