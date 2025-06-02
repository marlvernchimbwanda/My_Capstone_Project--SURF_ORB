# This the Monocular Visual odometry Package 


**1. To Run the #SURF Detector use:**

    `ros2 run monocular_visual_odom surf_detector` or just run the raw python script directly

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


## *How To Use the FILES*

1. **surf_orb_monocular_visual_odom.py:** it contains the code for the SURF-ORB Monocular Visual algorithm. Run this and it will show you the Trajectories and Plot of the Error per frame

2. **Orb_visual_odometry.py:** Is the ORB Version Which was used for comaprison with the SURF_ORB algorithm.


