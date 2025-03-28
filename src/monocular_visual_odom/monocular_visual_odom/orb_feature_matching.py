import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
import os
import time 
import psutil

# Function to measure memory usage
def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 ** 2)  # Convert to MB

hessian_threshold = 100 #find the hessian threshhold which will help us determine how much to keep
surf = cv.xfeatures2d.SURF_create(hessian_threshold, nOctaves=4, nOctaveLayers = 3)
orb = cv.ORB_create(hessian_threshold, scaleFactor=1.2,nlevels=4)

root = os.getcwd()
img1_path = os.path.join(root, 'datasets/KITTI_sequence_1/image_l/000008.png')
img2_path = os.path.join(root, 'datasets/KITTI_sequence_1/image_l/000009.png')
img1 = cv.imread('/home/uozrobotics/computer_vison_ros2/robovison_ros2_ws/src/robovision_ros2/data/images/mariachi.jpg', cv.IMREAD_GRAYSCALE)
img2 = cv.imread('/home/uozrobotics/computer_vison_ros2/robovison_ros2_ws/src/robovision_ros2/data/images/mariachi.jpg', cv.IMREAD_GRAYSCALE)

# Measure memory before processing
mem_before_surf_orb = get_memory_usage()
start_time = time.time()

# Detect keypoints with ORB and Compute Using ORB
keypoints1,descriptors1 = orb.detectAndCompute(img1,None)
keypoints2, descriptors2 = orb.detectAndCompute(img2,None)

# Measure memory after processing
orb_time = time.time() - start_time
mem_after_orb = get_memory_usage()

mem_usage_surf = mem_after_orb - mem_before_surf_orb



# def bruteForce():
   
    # orb = cv.ORB_create(nfeatures=800, scaleFactor=1.2,nlevels=4)
    # hessian_threshold = 8000 #find the hessian threshhold which will help us determine how much to keep
    # surf = cv.xfeatures2d.SURF_create(hessian_threshold)

    # start_time = time.time()
    # keypoints1,descriptor1 = orb.detectAndCompute(img1,None)
    # keypoints2,descriptors2 = orb.detectAndCompute(img2,None)
    # #use bruteforce matching object
    # # keypoints1 = surf.detect(img1)
    # # keypoints2 = surf.detect(img2)
    # # keypoints1, descriptor1 = orb.compute(img1, keypoints1)
    # # keypoints2, descriptors2 = orb.compute(img2, keypoints2)

    # bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
    # matches = bf.match(descriptor1,descriptors2)
    # matches = sorted(matches,key=lambda x:x.distance)
    # nMatches = 20
    # imgMatch = cv.drawMatches(img1,keypoints1,img2,keypoints2,matches[:nMatches], 
    #                           None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    # plt.figure('ORB Bruteforce Matching')
    # plt.imshow(imgMatch)
    # print(f"SURF time: {time.time() - start_time}")
    # plt.show()

def KnnBruteForce():
    
    

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck= False)
    nNeighbors = 2
    matches = bf.knnMatch(descriptors1,descriptors2, k=nNeighbors)
    good_Matches = []
    test_ratio = 0.75
    for m, n in matches:
        if m.distance < test_ratio*n.distance:
            good_Matches.append([m])

    inlier_ratio = len(good_Matches) / len(matches)

    img_match = cv.drawMatchesKnn(img1,keypoints1,img2,keypoints2,good_Matches,None
                                  ,flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    


        # Display results
    print(f"Standalone ORB Time: {orb_time:.4f} seconds")
    print(f"Standalone ORB Keypoints: {len(keypoints1)}")
    print(f"Standalone ORB Matches: {len(good_Matches)}")
    print(f"Standalone ORB Memory Usage: {mem_usage_surf:.4f} MB")
    print(f"Inlier Ratio: {inlier_ratio}")


    # Plot efficiency comparisons
    fig, axs = plt.subplots(2, 2, figsize=(7, 5))

        # Keypoints comparison
    axs[0, 0].bar(["Stanalone ORB"], [len(keypoints1)], color=['blue'])
    axs[0, 0].set_title("Number of Keypoints Detected")

    # Time comparison
    axs[0, 1].bar(["Standalone ORB"], [orb_time], color=['purple'])
    axs[0, 1].set_title("Time Taken (seconds)")

    # Matches comparison
    axs[1, 0].bar(["Standalone ORB"], [len(good_Matches)], color=['orange'])
    axs[1, 0].set_title("Number of Matches Found")

    # Memory comparison
    axs[1, 1].bar(["Standalone ORB"], [mem_usage_surf], color=['brown'])
    axs[1, 1].set_title("Memory Usage (MB)")

    plt.figure("ORB Matchimg using KNN Bruteforce")
    plt.imshow(img_match)
    plt.title("ORB Detector + ORB Descriptor")

    plt.show()

# def FLANN():

    # hessian_threshold = 8000 #find the hessian threshhold which will help us determine how much to keep
    # surf = cv.xfeatures2d.SURF_create(hessian_threshold)
    
    # keypoints1,descriptors1 = surf.detectAndCompute(img1,None)
    # keypoints2, descriptors2 = surf.detectAndCompute(img2,None)
    
    # flann_index_kdtree = 1
    # nkd_trees = 5 
    # nleaf_checks = 50
    # nNeighbours = 2
    # index_params = dict(algorithm = flann_index_kdtree, trees = nkd_trees)
    # search_params = dict(checks = nleaf_checks)

    # flann = cv.FlannBasedMatcher(index_params,search_params)
    # matches = flann.knnMatch(descriptors1,descriptors2,k=nNeighbours)
    # matchesMask = [[0,0] for i in range (len(matches))]
    # test_ratio = 0.75
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < test_ratio*n.distance:
    #          matchesMask[i] = [1,0]
    # draw_params = dict(matchColor=(0,255,0),singlePointColor=(255,0,0),
    #                    matchesMask=matchesMask,flags=cv.DrawMatchesFlags_DEFAULT)
    # img_match = cv.drawMatchesKnn(img1,keypoints1,img2,keypoints2, matches,None,
    #                               **draw_params)
    # plt.figure("KNN Feature Matching")
    # plt.imshow(img_match)
    # plt.show()

def main ():
    KnnBruteForce()



if __name__ == '__main__':
    main()