cmake_minimum_required(VERSION 3.16)
project(kas_utils)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(pybind11 REQUIRED)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS})

add_library(depth_to_point_cloud SHARED src/depth_to_point_cloud.cpp)
target_link_libraries(depth_to_point_cloud
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${pybind11_LIBRARIES})

pybind11_add_module(py_depth_to_point_cloud
    src/depth_to_point_cloud.cpp
    py_wrap/py_wrap_depth_to_point_cloud.cpp)
target_link_libraries(py_depth_to_point_cloud PRIVATE
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${pybind11_LIBRARIES})
