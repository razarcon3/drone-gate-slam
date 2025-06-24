# Graph-Based SLAM for Autonomous Drone Racing

[cite_start]This repository contains the source code and documentation for the senior thesis "Graph-Based SLAM for Autonomous Drone Racing" by Robert Azarcon from the University of Illinois at Urbana-Champaign. [cite_start]This project proposes and evaluates a Simultaneous Localization and Mapping (SLAM) system designed for autonomous drone racing, utilizing a factor graph optimization framework. Currently it only runs in offline mode.

## About The Project

[cite_start]The goal of this research is to enable a drone equipped with only a camera and an IMU sensor to localize itself and map the positions of gates on a race course. [cite_start]The system was developed and tested in a custom simulation environment using Unreal Engine 5.

### Key Features:
* [cite_start]**Gate Detection**: A Convolutional Neural Network (Faster R-CNN) is used to detect the corner points of race gates from the drone's camera feed.
* [cite_start]**Visual-Inertial Odometry (VIO)**: A tightly coupled VIO module (VINS-FUSION) runs in parallel to generate odometry data.
* [cite_start]**Factor Graph Optimization**: Gate corner detections and odometry are integrated over time in a factor graph to simultaneously estimate the drone's trajectory and the 3D positions of the gates.

## System Design

[cite_start]The SLAM system is composed of a "Front End" and a "Back End".

### Front End
The Front End processes visual data to identify and track features on the race gates.

1.  [cite_start]**Gate Corner Detection**: Faster R-CNN, trained on the 2019 AlphaPilot Challenge dataset, detects gate corners.
2.  [cite_start]**Corner Point Refinement**: A refinement step uses the Hough transform to find the precise intersection of the inner edges of the gate corners within the detected bounding boxes.
3.  [cite_start]**Gate Registration and Tracking**: The 3D points of the corners are geometrically verified to match the known dimensions of the gates. [cite_start]The verified corners are then tracked across frames using ByteTrack.

### Back End
[cite_start]The Back End constructs and solves a factor graph to estimate the drone's pose and gate locations.

* [cite_start]**Visual Projection Factors**: These factors constrain the estimated 3D corner positions to the current drone pose.
* [cite_start]**Ranging Factors**: The known physical dimensions of the gates are used to add rigid constraints between the estimated corner positions.
* [cite_start]**Between Factors**: Odometry information from VINS-FUSION is used to constrain temporally adjacent drone poses.

## Implementation

[cite_start]The system is implemented in Python 3.10 and relies on the following software packages:

| Software | Version | Purpose |
| :--- | :--- | :--- |
| NumPy | 1.26.4 | [cite_start]Array algorithms and frame transformations  |
| GTSAM | 4.2 | [cite_start]Factor graph modeling and optimization  |
| OpenCV | 4.11.0 | [cite_start]Image filtering, transformations, Hough transform, and non-maximum suppression  |
| PyTorch | 2.6.0 | [cite_start]FasterRCNN training and inference  |
| ByteTrack | 1.0 | [cite_start]Bounding box tracking  |
| VINS-FUSION | 1.0 | [cite_start]Visual Inertial Odometry (VIO)  |
| Cosys-AirSim | 3.2.0 | [cite_start]Drone Simulation in Unreal Engine 5  |

### Simulation Environment

[cite_start]The experiments were conducted in a modified version of the Unreal Engine 5's blocks world using the Cosys-AirSim simulator. [cite_start]The environment was set up with three drone gates placed at varying distances. [cite_start]The drone flew a straight-line trajectory towards the third gate.

## Results

[cite_start]The proposed SLAM system successfully estimated the drone's position and the location of the gates.

* [cite_start]The inclusion of gates as SLAM landmarks significantly reduced the positional drift compared to the VIO-only solution.
* [cite_start]The average positional error of the drone was reduced from 0.65 meters (VINS-FUSION only) to 0.27 meters with the proposed SLAM system. [cite_start]This is supported by the root mean square error (RMSE) of the L2 norm, which was 0.6576m for VIO and 0.2764m for the proposed SLAM.
* [cite_start]The estimated positions of the gates were determined with an average error of approximately 0.42 meters.
* [cite_start]There was a slight increase in the average angular error of about 0.35 degrees when using the proposed SLAM method.

## Citation

If you use this work, please cite the original thesis:

Azarcon, R. (2025). *Graph-Based SLAM for Autonomous Drone Racing* (Bachelor's thesis). University of Illinois at Urbana-Champaign.
