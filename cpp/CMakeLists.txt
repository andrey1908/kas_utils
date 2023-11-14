cmake_minimum_required(VERSION 3.16)
project(kas_utils_cpp)

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

# Install package.xml for catkin
install(FILES package.xml DESTINATION share/kas_utils_cpp/)

set(CONF_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
set(KAS_UTILS_CPP_CMAKE_DIR share/kas_utils_cpp/cmake)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  kas_utils_cpp-config.cmake.in
  ${PROJECT_BINARY_DIR}/cmake/kas_utils_cpp/kas_utils_cpp-config.cmake
  PATH_VARS KAS_UTILS_CPP_CMAKE_DIR
  INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/share/kas_utils_cpp
)

install(
  FILES ${PROJECT_BINARY_DIR}/cmake/kas_utils_cpp/kas_utils_cpp-config.cmake
  DESTINATION share/kas_utils_cpp/
)

install(
  TARGETS depth_to_point_cloud py_depth_to_point_cloud
  LIBRARY DESTINATION lib
)

