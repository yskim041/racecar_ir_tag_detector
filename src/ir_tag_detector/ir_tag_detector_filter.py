#!/usr/bin/env python

import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

from PIL import Image as PILImage

from ir_tag_detector_base import IRTagDetectorBase


class IRTagDetectorFilter(IRTagDetectorBase):
    def __init__(self, topic_name='/camera/ir/image_raw'):
        super(IRTagDetectorFilter, self).__init__()

        self.title = 'ir_tag_detector_filter'

        # self.init_ros_node(topic_name)

    def detect(self):
        if self.ir_msg is None:
            print('[IRTagDetector] Error: No input image')
            return

        mask = self.ir_msg
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        mask = cv2.bilateralFilter(mask, 30, 60, 60)
        mask[mask < np.max([80, np.max(mask) * 0.5])] = 0
        mask[mask > 0] = 255
        return mask


def test():
    print('Test IRTagDetectorFilter')

    detector = IRTagDetectorFilter()

    save_as_figure = True
    save_dir = 'rst'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_dir = 'Workspace/racecar_ws/samples/samples_1'
    for img_id in range(95):
        test_img_path = os.path.join(
            os.path.expanduser('~'),
            test_dir,
            'sample_ir_%04d.jpg' % img_id)
        detector.load_test_image(test_img_path)

        mask = detector.detect()

        if save_as_figure:
            fig, axarr = plt.subplots(1, 2, figsize=(20, 8))
            axarr[0].imshow(detector.ir_msg, cmap='gray')
            axarr[0].axis('off')
            axarr[1].imshow(mask, cmap='gray')
            axarr[1].axis('off')
            fig.tight_layout()
            save_filename = os.path.join(save_dir, 'rst_%04d.png' % img_id)
            fig.savefig(save_filename)
            print('[IRTagDetector] %s' % save_filename)
        else:
            save_img_path = os.path.join(save_dir, 'rc_%04d_ir.jpg' % img_id)
            out = PILImage.fromarray(detector.ir_msg.copy())
            out.save(save_img_path)
            print('[IRTagDetector] %s' % save_img_path)
            save_map_path = os.path.join(save_dir, 'rc_%04d_map.jpg' % img_id)
            out = PILImage.fromarray(mask.copy())
            out.save(save_map_path)
            print('[IRTagDetector] %s' % save_map_path)


if __name__ == '__main__':
    test()

# End of script
