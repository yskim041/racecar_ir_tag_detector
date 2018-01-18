# Racecar IR Tag Detector
The IR Tag Detector for Racecars


## Dependencies

Tested with
- Python 2.7.6
- OpenCV 3.2
- Tensorflow 1.4.0
- Pillow
- ROS Indigo

Check this repo if you want to install Tensorflow on the Jetson board:
https://github.com/jetsonhacks/installTensorFlowTX2

## IR Tag Detectors

1. Test the detector with simple filters
```
python src/ir_tag_detector/ir_tag_detector_filter.py
```

2. Test the detector with CNNs
```
python src/ir_tag_detector/ir_tag_detector_cnn.py
```
By running this script, you can train the networks. Try `detect()` method with an input image after finishing training.


## Image Collector
```
python src/ir_tag_detector/image_collector.py
```
This script collects RGB and IR images from zr300 sensor on a racecar.
