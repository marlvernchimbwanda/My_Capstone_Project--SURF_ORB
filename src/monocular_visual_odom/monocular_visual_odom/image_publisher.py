import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv 
import os


class ImagePublisher(Node):

    def __init__(self, img_path):
        super().__init__("image_publisher")

        #publishers
        self.img_publisher = self.create_publisher(Image, 'camera/image', 10)
       
        #read the image from the file 
        self.cv_img = cv.imread(img_path)

       #Convert the output data into a ROS message format
        self.bridge =CvBridge()
        self.image_message = self.bridge.cv2_to_imgmsg(self.cv_img, "passthrough")
        timer_period = 1.0
        self.timer = self.create_timer(timer_period,self.timer_callback)

    def timer_callback(self):
        self.img_publisher.publish(self.image_message)
        self.get_logger().info("Publishing the image in python")


def main(args=None):
    rclpy.init(args=args)
    root = os.getcwd()
    img1_path = os.path.join('/home/uozrobotics/capstone1_ws/src/datasets/KITTI_sequence_2/image_l/000000.png' )
    node = ImagePublisher(img1_path)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
