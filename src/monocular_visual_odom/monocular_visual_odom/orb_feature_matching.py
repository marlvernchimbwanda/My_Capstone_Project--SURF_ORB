import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
import os

root = os.getcwd()
img1_path = os.path.join(root, 'datasets/KITTI_sequence_1/image_l/000008.png')
img2_path = os.path.join(root, 'datasets/KITTI_sequence_1/image_l/000009.png')
img1 = cv.imread(img1_path)
img2 = cv.imread(img2_path)

def bruteForce():
   
    orb = cv.ORB_create()
    hessian_threshold = 8000 #find the hessian threshhold which will help us determine how much to keep
    surf = cv.xfeatures2d.SURF_create(hessian_threshold)

    keypoints1,descriptor1 = orb.detectAndCompute(img1,None)
    keypoints2,descriptors2 = orb.detectAndCompute(img2,None)
    #use bruteforce matching object
    # keypoints1 = surf.detect(img1)
    # keypoints2 = surf.detect(img2)
    # keypoints1, descriptor1 = orb.compute(img1, keypoints1)
    # keypoints2, descriptors2 = orb.compute(img2, keypoints2)

    bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
    matches = bf.match(descriptor1,descriptors2)
    matches = sorted(matches,key=lambda x:x.distance)
    nMatches = 20
    imgMatch = cv.drawMatches(img1,keypoints1,img2,keypoints2,matches[:nMatches], 
                              None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure('Matching')
    plt.imshow(imgMatch)
    plt.show()

def KnnBruteForce():
    hessian_threshold = 8000 #find the hessian threshhold which will help us determine how much to keep
    surf = cv.xfeatures2d.SURF_create(hessian_threshold)
    
    keypoints1,descriptors1 = surf.detectAndCompute(img1,None)
    keypoints2, descriptors2 = surf.detectAndCompute(img2,None)

    bf = cv.BFMatcher()
    nNeighbors = 2
    matches = bf.knnMatch(descriptors1,descriptors2, k=nNeighbors)
    good_Matches = []
    test_ratio = 0.75
    for m, n in matches:
        if m.distance < test_ratio*n.distance:
            good_Matches.append([m])

    img_match = cv.drawMatchesKnn(img1,keypoints1,img2,keypoints2,good_Matches,None
                                  ,flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    

    plt.figure("SURF Matchimg using KNN")
    plt.imshow(img_match)
    plt.show()

def FLANN():

    hessian_threshold = 8000 #find the hessian threshhold which will help us determine how much to keep
    surf = cv.xfeatures2d.SURF_create(hessian_threshold)
    
    keypoints1,descriptors1 = surf.detectAndCompute(img1,None)
    keypoints2, descriptors2 = surf.detectAndCompute(img2,None)
    
    flann_index_kdtree = 1
    nkd_trees = 5 
    nleaf_checks = 50
    nNeighbours = 2
    index_params = dict(algorithm = flann_index_kdtree, trees = nkd_trees)
    search_params = dict(checks = nleaf_checks)

    flann = cv.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors1,descriptors2,k=nNeighbours)
    matchesMask = [[0,0] for i in range (len(matches))]
    test_ratio = 0.75
    for i,(m,n) in enumerate(matches):
        if m.distance < test_ratio*n.distance:
             matchesMask[i] = [1,0]
    draw_params = dict(matchColor=(0,255,0),singlePointColor=(255,0,0),
                       matchesMask=matchesMask,flags=cv.DrawMatchesFlags_DEFAULT)
    img_match = cv.drawMatchesKnn(img1,keypoints1,img2,keypoints2, matches,None,
                                  **draw_params)
    plt.figure("KNN Feature Matching")
    plt.imshow(img_match)
    plt.show()



if __name__ == '__main__':
     bruteForce()
    # KnnBruteForce()
   # FLANN()