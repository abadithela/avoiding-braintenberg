cmake_minimum_required(VERSION 2.8.3)
project(avoiding_brightness)

find_package(catkin REQUIRED COMPONENTS
  rospy
  duckietown_msgs
  std_msgs
  sensor_msgs
)

catkin_package()
