#!/usr/bin/env python

import os
import time
import rospy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage


image_topic_name = '/camera/color/image_raw'
img_msg = None
img_counter = 0

ir_topic_name = '/camera/ir/image_raw'
ir_msg = None

save_dir = 'samples'

cvbridge = CvBridge()


def sensor_image_callback(ros_data):
    global img_msg
    img_msg = cvbridge.imgmsg_to_cv2(ros_data, ros_data.encoding)


def sensor_ir_callback(ros_data):
    global ir_msg
    ir_msg = cvbridge.imgmsg_to_cv2(ros_data, ros_data.encoding)


def save_image():
    if img_msg is not None and ir_msg is not None:
        save_img_path = os.path.join(
            save_dir, 'sample_img_%04d.jpg' % img_counter)
        save_ir_path = os.path.join(
            save_dir, 'sample_ir_%04d.jpg' % img_counter)
        print(img_msg.shape, ir_msg.shape)
        img = PILImage.fromarray(img_msg.copy())
        img.save(save_img_path)
        print(save_img_path)
        img = PILImage.fromarray(ir_msg.copy())
        img.save(save_ir_path)
        print(save_ir_path)
        global img_counter
        img_counter += 1
    else:
        print('no image')
    time.sleep(1)


def script_main():
    print('collecting images from zr300')

    rospy.init_node('image_collector')
    rospy.Subscriber(
        image_topic_name, Image,
        sensor_image_callback, queue_size=1)
    print('subscribed to %s' % image_topic_name)

    rospy.Subscriber(
        ir_topic_name, Image,
        sensor_ir_callback, queue_size=1)
    print('subscribed to %s' % ir_topic_name)

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for _ in range(10000):
        save_image()


if __name__ == '__main__':
    script_main()

# End of script
