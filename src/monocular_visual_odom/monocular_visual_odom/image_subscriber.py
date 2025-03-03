#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import traceback

class ImageSubscriberNode(Node):
    def __init__(self):
        super().__init__("image_subscriber")

        self.img = []
        self.scaled_img = []
        self.is_img = False
        self.is_scaled_image = False
        self.display = True

        #subscribers
        self.img_subscriber = self.create_subscription(Image, "camera/image", self.camera_image_callback,10)

        #Processing
        self.proocessing_timer = self.create_timer(0.030, self.image_processing) #update each image 30 milliseconds 


    def camera_image_callback(self, msg):
        '''
        Here we are reading the ROS Message and convert it back to CV2 readable formart
        '''
        try:
           # transform the new message to an OpenCv Matrix
           self.bridge = CvBridge()
           self.img = self.bridge.imgmsg_to_cv2(msg, msg.encoding).copy() #converting the ros message to open cv compatible image
           self.is_img = True

        except:
            self.get_logger().error(traceback.format_exc())


    def image_processing(self):
        '''
        Here we are processing the image which we have read
        '''
        if self.is_img:
            #show the images
            if (self.display):
                cv2.imshow('view', self.img)
                cv2.waitKey(1)
            



def main (args = None):
    rclpy.init(args=args)
    node = ImageSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()