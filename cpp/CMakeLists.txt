cmake_minimum_required(VERSION 3.16)
project(kas_utils_cpp)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV)
find_package(PCL)
find_package(yaml-cpp)
find_package(CGAL)
find_package(pybind11)

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${CGAL_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

add_library(kas_utils SHARED src/utils.cc)  # use name "kas_utils" to avoid conflicts when used as subdirectory in another projects
add_executable(collection_test src/collection_test.cc)
target_link_libraries(collection_test kas_utils)
add_executable(stamped_collection_test src/stamped_collection_test.cc)
target_link_libraries(stamped_collection_test kas_utils)
add_executable(time_measurer_test src/time_measurer_test.cc)
target_link_libraries(time_measurer_test kas_utils)

if (${OPENCV_FOUND} AND ${PCL_FOUND})
    add_library(depth_to_point_cloud SHARED src/depth_to_point_cloud.cc)
    target_link_libraries(depth_to_point_cloud
        ${OpenCV_LIBS}
        ${PCL_LIBRARIES}
    )

    if (${pybind11_FOUND})
        pybind11_add_module(py_depth_to_point_cloud
            src/depth_to_point_cloud.cc
            py_wrap/py_wrap_depth_to_point_cloud.cc
        )
        target_link_libraries(py_depth_to_point_cloud PRIVATE
            ${OpenCV_LIBS}
            ${PCL_LIBRARIES}
            ${pybind11_LIBRARIES}
        )
    endif()
endif()

if (${OPENCV_FOUND})
    add_library(dilation SHARED src/dilation.cc)
    target_link_libraries(dilation
        ${OpenCV_LIBS}
    )
endif()

if (${yaml-cpp_FOUND})
    add_library(yaml_utils SHARED src/yaml_utils.cc)
    target_link_libraries(yaml_utils
        ${YAML_CPP_LIBRARIES}
    )
endif()

if (${CGAL_FOUND})
    add_library(cgal_utils SHARED src/cgal_utils.cc)
    target_link_libraries(cgal_utils
        ${CGAL_LIBRARIES}
    )
endif()

include(CMakePackageConfigHelpers)
set(INCLUDE_DIR "include")
set(LIBRARY_DIR "lib")
configure_package_config_file(
    kas_utils_cpp-config.cmake.in
    ${PROJECT_BINARY_DIR}/cmake/kas_utils_cpp/kas_utils_cpp-config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_PREFIX}/share/kas_utils_cpp
    PATH_VARS INCLUDE_DIR LIBRARY_DIR
)

install(
    FILES package.xml
    DESTINATION share/kas_utils_cpp
)

install(
    FILES ${PROJECT_BINARY_DIR}/cmake/kas_utils_cpp/kas_utils_cpp-config.cmake
    DESTINATION share/kas_utils_cpp
)

install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/kas_utils
    DESTINATION include
)

install(
    TARGETS kas_utils
    LIBRARY DESTINATION lib
)

if (TARGET depth_to_point_cloud)
    install(
        TARGETS depth_to_point_cloud
        LIBRARY DESTINATION lib
    )
endif()

if (TARGET py_depth_to_point_cloud)
    install(
        TARGETS py_depth_to_point_cloud
        LIBRARY DESTINATION lib/python3/dist-packages
    )
endif()

if (TARGET dilation)
    install(
        TARGETS dilation
        LIBRARY DESTINATION lib
    )
endif()

if (TARGET yaml_utils)
    install(
        TARGETS yaml_utils
        LIBRARY DESTINATION lib
    )
endif()

if (TARGET cgal_utils)
    install(
        TARGETS cgal_utils
        LIBRARY DESTINATION lib
    )
endif()
