# This the Monocular Visual odometry Package 


**1. To Run the #SURF Detector use:**

    `ros2 run monocular_visual_odom surf_detector`

--------

**2. To Run the #ORB detector use:**

    `ros2 run monocular_visual_odom orb_detector` 

---

**3. To Run the #ORB feature detector and matching node use:**

    `ros2 run monocular_visual_odom orb_feature_matching`

---

- The *orb_detetctor.py* is the node for the orb detector. 
- The  *surf_detector* is the node for the surf detector
- The *orb_feature_matching.py* is the raw python code for the orb feature matching algorithm.
- The *ros_orb_fetaure matyching.py* is the node for the orb feature matching which i was trying to put to ros 2. 

