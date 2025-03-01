### About the Project

This project focuses on improving **scale invariance** in the **Oriented FAST and Rotated BRIEF (ORB)** algorithm for **Visual Odometry (VO)** applications. ORB is a fast and efficient feature detection and description algorithm, but it lacks robust scale invariance, which is crucial for applications like autonomous navigation, robotics, and augmented reality.

To address this limitation, the project combines the strengths of **Speeded-Up Robust Features (SURF)**—a scale-invariant feature detector—with ORB's efficient descriptor. The resulting **hybrid SURF-ORB algorithm** aims to enhance feature detection and tracking across varying scales, making it more robust for real-world applications.

The project involves:
- **Feature Detection**: Using SURF to detect scale-invariant keypoints.
- **Feature Description**: Using ORB to describe these keypoints efficiently.
- **Motion Estimation**: Estimating camera motion by matching features across consecutive frames.

The algorithm is implemented in **Python** using **OpenCV** and tested on benchmark datasets to evaluate its accuracy, robustness, and computational efficiency. This project contributes to advancing visual odometry by addressing a key limitation in ORB, making it more suitable for real-time applications in dynamic environments.
