cmake_minimum_required(VERSION 2.8.3)
project(ir_tag_detector)

find_package(catkin REQUIRED)
catkin_package()
catkin_python_setup()

install(
    PROGRAMS
        image_collector.py
        ir_tag_detector_filter.py
        ir_tag_detector_cnn.py
    DESTINATION
        "${CATKIN_PACKAGE_BIN_DESTINATION}"
)
