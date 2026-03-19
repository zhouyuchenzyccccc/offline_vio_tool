cmake_minimum_required(VERSION 3.10)
project(orbslam3_offline_runner)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ORB_SLAM3_ROOT "" CACHE PATH "Path to ORB-SLAM3 root")
if(NOT ORB_SLAM3_ROOT)
  message(FATAL_ERROR "Please set ORB_SLAM3_ROOT to your ORB-SLAM3 source path")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

include_directories(
  ${ORB_SLAM3_ROOT}
  ${ORB_SLAM3_ROOT}/include
  ${ORB_SLAM3_ROOT}/include/CameraModels
  ${ORB_SLAM3_ROOT}/Thirdparty/Sophus
  ${EIGEN3_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
)

find_library(ORB_SLAM3_LIB
  NAMES ORB_SLAM3 libORB_SLAM3
  PATHS
    ${ORB_SLAM3_ROOT}/lib
    ${ORB_SLAM3_ROOT}/build/lib
    ${ORB_SLAM3_ROOT}/build
)

if(NOT ORB_SLAM3_LIB)
  message(FATAL_ERROR "Cannot find ORB_SLAM3 library. Build ORB-SLAM3 first.")
endif()

add_executable(rgbd_inertial_offline rgbd_inertial_offline.cc)

target_link_libraries(rgbd_inertial_offline
  ${ORB_SLAM3_LIB}
  ${OpenCV_LIBS}
  OpenGL::GL
  GLEW::GLEW
)

if(UNIX)
  target_link_libraries(rgbd_inertial_offline pthread)
endif()

message(STATUS "Using ORB_SLAM3_LIB=${ORB_SLAM3_LIB}")
