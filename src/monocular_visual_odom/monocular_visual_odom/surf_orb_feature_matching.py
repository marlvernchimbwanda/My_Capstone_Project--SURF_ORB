import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
import os
import time
import psutil

# Function to measure memory usage
def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 ** 2)  # Convert to MB


hessian_threshold = 3390 #find the hessian threshhold which will help us determine how much to keep
surf = cv.xfeatures2d.SURF_create(hessian_threshold, nOctaves=4, nOctaveLayers = 3)
orb = cv.ORB_create(hessian_threshold, scaleFactor=1.2,nlevels=4)
    
root = os.getcwd()
img1_path = os.path.join(root, 'datasets/KITTI_sequence_1/image_l/000008.png')
img2_path = os.path.join(root, 'datasets/KITTI_sequence_1/image_l/000009.png')
img1 = cv.imread ('/home/uozrobotics/computer_vison_ros2/robovison_ros2_ws/src/robovision_ros2/data/images/mariachi.jpg') #)('/home/uozrobotics/computer_vison_ros2/robovison_ros2_ws/src/robovision_ros2/data/images/baboon.png',cv.IMREAD_GRAYSCALE)
img2 = cv.imread ('/home/uozrobotics/computer_vison_ros2/robovison_ros2_ws/src/robovision_ros2/data/images/mariachi.jpg') #('/home/uozrobotics/computer_vison_ros2/robovison_ros2_ws/src/robovision_ros2/data/images/baboon2.png', cv.IMREAD_GRAYSCALE)

# Measure memory before processing
mem_before_surf_orb = get_memory_usage()

start_time = time.time()

# Detect keypoints with SURF
start_time = time.time()
kp1_surf = surf.detect(img1, None)
kp2_surf = surf.detect(img2, None)

# Compute ORB descriptors
kp1_surf, des1_orb = orb.compute(img1, kp1_surf)
kp2_surf, des2_orb = orb.compute(img2, kp2_surf)

# Measure memory after processing
surf_orb_time = time.time() - start_time
mem_after_surf_orb = get_memory_usage()

mem_usage_surf = mem_after_surf_orb - mem_before_surf_orb



def KnnBruteForce():
    
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    nNeighbors = 2
    matches = bf.knnMatch(des1_orb,des2_orb, k=nNeighbors)
    good_Matches = []
    test_ratio = 0.75 #Apply Lowe's ratio test toremove outliers 
    for m, n in matches:
        if m.distance < test_ratio*n.distance:
            good_Matches.append([m])

    inlier_ratio = len(good_Matches) / len(matches)

    img_match = cv.drawMatchesKnn(img1,kp1_surf,img2,kp2_surf,good_Matches,None
                                  ,flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    

        # Display results
    print(f"SURF+ORB Time: {surf_orb_time:.4f} seconds")
    print(f"SURF+ORB Keypoints: {len(kp1_surf)}")
    print(f"SURF+ORB Matches: {len(good_Matches)}")
    print(f"SURF+ORB Memory Usage: {mem_usage_surf:.4f} MB")
    print(f"Inlier Ratio: {inlier_ratio}")
    # Plot efficiency comparisons
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))

        # Keypoints comparison
    axs[0, 0].bar(["SURF+ORB"], [len(kp1_surf)], color=['blue'])
    axs[0, 0].set_title("Number of Keypoints Detected")

    # Time comparison
    axs[0, 1].bar(["SURF+ORB"], [surf_orb_time], color=['purple'])
    axs[0, 1].set_title("Time Taken (seconds)")

    # Matches comparison
    axs[1, 0].bar(["SURF+ORB"], [len(good_Matches)], color=['orange'])
    axs[1, 0].set_title("Number of Matches Found")

    # Memory comparison
    axs[1, 1].bar(["SURF+ORB"], [mem_usage_surf], color=['brown'])
    axs[1, 1].set_title("Memory Usage (MB)")

    plt.figure("SURF Matchimg using KNN Bruteforce")
    plt.imshow(img_match)
    plt.title("SURF Detector + ORB Descriptor")

    plt.show()


def main ():
    KnnBruteForce()



if __name__ == '__main__':
    main()