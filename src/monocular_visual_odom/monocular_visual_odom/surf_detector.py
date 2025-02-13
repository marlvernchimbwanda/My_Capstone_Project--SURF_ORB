'This Node Runs the Surf detector algorithm. '

import rclpy
from rclpy.node import Node

import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
import os

class MySurf(Node):

    def __init__(self):
        super().__init__("surf_detector")
        self.root = os.getcwd()
        self.img_path = os.path.join(self.root, 'datasets/KITTI_sequence_1/image_l/000000.png')
        self.img_gray = cv.imread(self.img_path) 

        self.hessian_threshold = 8000 #find the hessian threshhold which will help us determine how much to keep
        self.surf = cv.xfeatures2d.SURF_create(self.hessian_threshold)
        self.keypoints = self.surf.detect(self.img_gray)
        self.img_gray = cv.drawKeypoints(self.img_gray,self.keypoints, self.img_gray,flags=cv.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        plt.figure()
        plt.imshow(self.img_gray)
        plt.show()



def main(args = None):
    rclpy.init(args=args) #initialize ros2 communications
    node = MySurf()
    rclpy.spin(Node)
    rclpy.shutdown() #Emd ros2 communiccations

if __name__ == '__main__':
    main()