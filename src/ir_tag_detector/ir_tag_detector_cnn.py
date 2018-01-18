#!/usr/bin/env python

import numpy as np
import os
import rospy
import tensorflow as tf

from sensor_msgs.msg import Image as ImageMsg
from cv_bridge import CvBridge
from PIL import Image as PILImage

from utils import tf_utils


class IRTagDetectorCNN:
    def __init__(self, topic_name='/camera/ir/image_raw'):
        self.title = 'ir_tag_detector_cnn'

        self.ir_msg = None
        self.cvbridge = CvBridge()

        self.data_ir = list()
        self.data_mask = list()

        self.feature_size = 1024
        self.batch_size = 1
        self.lr = 1e-4
        self.epoches = 100
        self.gpu_id = '0'
        self.checkout_dir = 'progress'

        self.sess = None
        self.init_tf_session()

        # self.init_ros_node(topic_name)

    def sensor_image_callback(self, ros_data):
        self.ir_msg = self.cvbridge.imgmsg_to_cv2(ros_data, ros_data.encoding)

    def init_ros_node(self, topic_name):
        rospy.init_node(self.title)
        rospy.Subscriber(
            topic_name, ImageMsg,
            self.sensor_image_callback, queue_size=1)
        print('[IRTagDetector] Subscribed to %s' % topic_name)

    def load_training_data(self, base_dir):
        print('load_training_data')
        for idx in range(0, 95):
            img_ir = PILImage.open(
                os.path.join(base_dir, 'rc_%04d_ir.jpg' % idx))

            img_mask = PILImage.open(
                os.path.join(base_dir, 'rc_%04d_mask.jpg' % idx))

            self.data_ir.append(np.array(img_ir)[:, :, None])
            self.data_mask.append(np.array(img_mask)[:, :, None])

    def load_test_image(self, filename):
        img = PILImage.open(filename)
        # img = img.resize(self.img_size, PILImage.ANTIALIAS)
        self.ir_msg = np.array(img)

    def init_tf_session(self):
        print('[IRTagDetector] init tf session')

        tf.reset_default_graph()

        self.input = tf.placeholder(
            tf.float32, shape=[self.batch_size, 360, 480, 1])
        self.mask = tf.placeholder(
            tf.float32, shape=[self.batch_size, 360, 480, 1])

        self.keep_prob = tf.placeholder(tf.float32)

        # cl0 (360, 480, 4)
        cl0 = tf_utils.conv(self.input, 1, 4, strides=[1, 1, 1, 1], ksize=3)
        # cl1 (180, 240, 8)
        cl1 = tf_utils.conv_down(cl0, 4, ksize=3)
        # cl2 (90, 120, 16)
        cl2 = tf_utils.conv_down(cl1, 8, ksize=3)
        # cl3 (45, 60, 32)
        cl3 = tf_utils.conv_down(cl2, 16, ksize=3)
        # cl4 (23, 30, 64)
        cl4 = tf_utils.conv_down(cl3, 32, ksize=3)
        # cl5 (12, 15, 128)
        cl5 = tf_utils.conv_down(cl4, 64, ksize=3)
        # cl6 (6, 8, 256)
        cl6 = tf_utils.conv_down(cl5, 128, ksize=3)

        fcl_down = tf_utils.conv_flat(cl6, 6 * 8 * 256, self.feature_size)

        self.fcl = tf.nn.relu(
            tf_utils.fc_layer(fcl_down, self.feature_size, self.feature_size))
        self.fcl = tf.nn.dropout(self.fcl, self.keep_prob)

        # dl6 (6, 8, 256)
        dl6 = tf_utils.deconv_flat(
            self.fcl, self.feature_size, [6, 8, 256], self.batch_size)
        # dl5 (12, 15, 128)
        dl5 = tf_utils.deconv_with_concat(
            dl6, cl5, 256, [self.batch_size, 12, 15, 128])
        # dl4 (23, 30, 64)
        dl4 = tf_utils.deconv_with_concat(
            dl5, cl4, 128, out_shape=[self.batch_size, 23, 30, 64])
        # dl3 (45, 60, 32)
        dl3 = tf_utils.deconv_with_concat(
            dl4, cl3, 64, out_shape=[self.batch_size, 45, 60, 32])
        # dl2 (90, 120, 16)
        dl2 = tf_utils.deconv_with_concat(
            dl3, cl2, 32, out_shape=[self.batch_size, 90, 120, 16])
        # dl1 (180, 240, 8)
        dl1 = tf_utils.deconv_with_concat(
            dl2, cl1, 16, out_shape=[self.batch_size, 180, 240, 8])
        # dl0 (360, 480, 4)
        dl0 = tf_utils.deconv_with_concat(
            dl1, cl0, 8, out_shape=[self.batch_size, 360, 480, 4])

        self.pred = tf_utils.deconv(
            dl0, 4, out_shape=[self.batch_size, 360, 480, 1],
            strides=[1, 1, 1, 1])

        self.loss = tf.reduce_mean(tf.losses.mean_squared_error(
            labels=self.mask, predictions=self.pred))

        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.visible_device_list = self.gpu_id
        tf_config.gpu_options.allow_growth = False
        self.sess = tf.Session(config=tf_config)

        self.saver = tf.train.Saver()

        checkpoint = tf.train.latest_checkpoint(self.checkout_dir)
        if checkpoint:
            print('Restoring from checkpoint: {}'.format(checkpoint))
            self.saver.restore(self.sess, checkpoint)
        else:
            print('Couldn\'t find checkpoint to restore from. Starting over.')
            self.sess.run(tf.global_variables_initializer())

    def train(self, base_dir='data'):
        print('train')

        self.load_training_data('data')

        bsize = self.batch_size
        max_rounds = int(len(self.data_ir) / bsize)

        for ei in range(self.epoches):
            for ri in range(max_rounds):
                batch_ir = self.data_ir[bsize * ri:bsize * ri + bsize]
                batch_mask = self.data_mask[bsize * ri:bsize * ri + bsize]

                _, loss, pred = self.sess.run(
                    [self.train_step, self.loss, self.pred],
                    feed_dict={self.input: batch_ir,
                               self.mask: batch_mask,
                               self.keep_prob: 0.9})

                if ri % 10 == 0:
                    print('ep %d, step %d, loss %.6f' % (ei, ri, loss))

                save_img_path = 'pred/rst_%03d.jpg' % ri
                out = PILImage.fromarray(np.squeeze(pred).copy())
                out = out.convert('RGB')
                out.save(save_img_path)

            self.saver.save(
                self.sess, os.path.join(self.checkout_dir, self.title))

    def detect(self, img):
        return self.sess.run(
            self.pred, feed_dict={self.input: img[None, :, :, None]})


def test():
    print('Test IRTagDetectorCNN')

    detector = IRTagDetectorCNN()
    detector.train()
    detector.sess.close()


if __name__ == '__main__':
    test()

# End of script
