### About the Project

This project focuses on improving **scale invariance** in the **Oriented FAST and Rotated BRIEF (ORB)** algorithm for **Visual Odometry (VO)** applications. ORB is a fast and efficient feature detection and description algorithm, but it lacks robust scale invariance, which is crucial for applications like autonomous navigation, robotics, and augmented reality.

To address this limitation, the project combines the strengths of **Speeded-Up Robust Features (SURF)**—a scale-invariant feature detector—with ORB's efficient descriptor. The resulting **hybrid SURF-ORB algorithm** aims to enhance feature detection and tracking across varying scales, making it more robust for real-world applications.

The project involves:
- **Feature Detection**: Using SURF to detect scale-invariant keypoints.
- **Feature Description**: Using ORB to describe these keypoints efficiently.
- **Motion Estimation**: Estimating camera motion by matching features across consecutive frames.

The algorithm is implemented in **Python** using **OpenCV** and tested on benchmark datasets to evaluate its accuracy, robustness, and computational efficiency. This project contributes to advancing visual odometry by addressing a key limitation in ORB, making it more suitable for real-time applications in dynamic environments.







# SURF–ORB: A Hybrid Visual Odometry Pipeline


This repository contains a **Monocular Visual Odometry (VO)** implementation using a hybrid of the **SURF feature detector** and the **ORB descriptor**. The project aims to improve scale invariance and trajectory consistency by combining the strengths of both algorithms.


---


## Overview


**Visual Odometry (VO)** refers to the process of estimating the position and orientation (*pose*) of a moving agent (e.g., robot or vehicle) using camera data. In this project:


- **SURF (Speeded-Up Robust Features)** is used to detect keypoints that are robust to scale and rotation.
- **ORB (Oriented FAST and Rotated BRIEF)** is used to compute efficient binary descriptors for matching.
- The combination (SURF–ORB) aims to mitigate the limitations of ORB in scale-variant environments while preserving speed.


---


## 🔍 Project Objectives


- ✅ Improve the **scale invariance** of ORB-based VO systems.
- ✅ Compare the hybrid SURF–ORB approach against standard ORB across multiple KITTI sequences.
- ✅ Evaluate performance in terms of:
 - Inlier ratios
 - Pose accuracy (trajectory vs ground truth)
 - Computation time (ms/frame)
 - Robustness in scale-changing environments


---


## 🧪 Datasets Used


- 🔹 [KITTI Odometry Dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) 
- Tested on 8 sequences (00–07), with varying conditions:
 - Open roads
 - Urban environments
 - Sharp turns and rotations


---


## Evaluation Metrics


- **Cumulative Trajectory Error**
- **Inlier Ratio (%)** after RANSAC
- **Frame Processing Time** (ms/frame)
- **Standard Deviation of Pose Error**
- **Visual Path Comparison** against Ground Truth


---


## REFERENCES


>- @inproceedings{Geiger2012CVPR,
 author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
 title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
 booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
 year = {2012}
}
