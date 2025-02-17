import rclpy
from rclpy.node import Node

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os


class MyOrb(Node):

    def __init__(self):
        super().__init__('orb_dtetctor')
        self.root = os.getcwd()
        self.img_path = os.path.join(self.root, 'src/datasets/KITTI_sequence_1/image_l/000000.png')
        self.img_gray = cv.imread(self.img_path)

        self.orb = cv.ORB_create()
        self.keypoints = self.orb.detect(self.img_gray)
        self.keypoints,_ = self.orb.compute(self.img_gray,self.keypoints)
        self.img_gray = cv.drawKeypoints(self.img_gray,self.keypoints,self.img_gray, 
                                         flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.figure()
        plt.imshow(self.img_gray)
        plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = MyOrb()
    rclpy.spin(node)
    rclpy.shutdown()



