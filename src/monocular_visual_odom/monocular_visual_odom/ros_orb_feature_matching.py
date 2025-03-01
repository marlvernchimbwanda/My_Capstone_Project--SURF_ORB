
import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
import os

import rclpy
from rclpy.node import Node

#to do 
#find out how we can run the indivisual functions in a node. eisthe use publisher or something
class FeatureMatching(Node):
    
    def __init__(self):
        super().__init__("orb_feature_matching")
        self.root = os.getcwd()
        self.img1_path = os.path.join(self.root, 'src/datasets/KITTI_sequence_1/image_l/000008.png')
        self.img2_path = os.path.join(self.root, 'src/datasets/KITTI_sequence_1/image_l/000009.png')
        self.img1 = cv.imread(self.img1_path)
        self.img2 = cv.imread(self.img2_path)

        self.orb = cv.ORB_create()
        self.hessian_threshold = 8000 #find the hessian threshhold which will help us determine how much to keep
        self.surf = cv.xfeatures2d.SURF_create(self.hessian_threshold)

        self.keypoints1,self.descriptor1 = self.orb.detectAndCompute(self.img1,None)
        self.keypoints2,self.descriptors2 = self.orb.detectAndCompute(self.img2,None)
        #use bruteforce matching object
        
        #surf orb
        # keypoints1 = surf.detect(self.img1)
        # keypoints2 = surf.detect(self.img2)
        # keypoints1, descriptor1 = orb.compute(self.img1, keypoints1)
        # keypoints2, descriptors2 = orb.compute(self.img2, keypoints2)

    #def bruteForce(self):
    
        self.bf = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True)
        self.matches = self.bf.match(self.descriptor1,self.descriptors2)
        self.matches = sorted(self.matches,key=lambda x:x.distance)
        self.nMatches = 1
        self.imgMatch = cv.drawMatches(self.img1,self.keypoints1,self.img2,self.keypoints2,self.matches[:self.nMatches], 
                                None, flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure('Bruteforce Matching')
        plt.imshow(self.imgMatch)
        plt.show()

    def KnnBruteForce(self):
       
        self.bf = cv.BFMatcher()
        self.nNeighbors = 2
        self.matches = self.bf.knnMatch(self.descriptors1,self.descriptors2, k=self.nNeighbors)
        self.good_Matches = []
        self.test_ratio = 0.75
        for m, n in self.matches:
            if m.distance < self.test_ratio*n.distance:
                self.good_Matches.append([m])

        self.img_match = cv.drawMatchesKnn(self.img1,self.keypoints1,self.img2,self.keypoints2,self.good_Matches,None
                                    ,flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
        

        plt.figure("ORB Matchimg using KNN BruteForce")
        plt.imshow(self.img_match)
        plt.show()

    def FLANN(self):
        
        self.flann_index_kdtree = 1
        self.nkd_trees = 5 
        self.nleaf_checks = 50
        self.nNeighbours = 2
        self.index_params = dict(algorithm = self.flann_index_kdtree, trees = self.nkd_trees)
        self.search_params = dict(checks = self.nleaf_checks)

        flann = cv.FlannBasedMatcher(self.index_params,self.search_params)
        matches = flann.knnMatch(self.descriptors1,self.descriptors2,k=self.nNeighbours)
        matchesMask = [[0,0] for i in range (len(matches))]
        self.test_ratio = 0.75
        for i,(m,n) in enumerate(matches):
            if m.distance < self.test_ratio*n.distance:
                matchesMask[i] = [1,0]
        draw_params = dict(matchColor=(0,255,0),singlePointColor=(255,0,0),
                        matchesMask=matchesMask,flags=cv.DrawMatchesFlags_DEFAULT)
        img_match = cv.drawMatchesKnn(self.img1,self.keypoints1,self.img2,self.keypoints2, matches,None,
                                    **draw_params)
        plt.figure("FLANN Feature Matching")
        plt.imshow(img_match)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = FeatureMatching()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
   # bruteForce()
  # KnnBruteForce()
  #FLANN()
     main()