# ir_tag_detector_base.py

import numpy as np
import rospy

from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge
from PIL import Image as PILImage


class IRTagDetectorBase(object):
    def __init__(self):
        self.ir_msg = None
        self.cvbridge = CvBridge()

    def sensor_image_callback(self, ros_data):
        self.ir_msg = self.cvbridge.imgmsg_to_cv2(ros_data, ros_data.encoding)

    def init_ros_node(self, topic_name):
        rospy.init_node(self.title)
        rospy.Subscriber(
            topic_name, ImageMsg,
            self.sensor_image_callback, queue_size=1)
        print('[IRTagDetector] Subscribed to %s' % topic_name)

    def load_test_image(self, filename):
        img = PILImage.open(filename)
        # img = img.resize([480, 360], PILImage.ANTIALIAS)
        self.ir_msg = np.array(img)


# End of script
