cmake_minimum_required(VERSION 3.16)
project(kas_utils_python)

find_package(kas_utils_cpp REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# build python package
add_custom_target(custom ALL
    COMMAND ${CMAKE_COMMAND} -E env kas_utils_cpp_PYTHON_LIBRARIES=${kas_utils_cpp_PYTHON_LIBRARIES}
        ${Python3_EXECUTABLE} setup.py develop --prefix develop
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

install(
    FILES package.xml
    DESTINATION share/${PROJECT_NAME}
)

# create symlink to package
install(
    CODE "
        file(MAKE_DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages)
        file(CREATE_LINK ${CMAKE_CURRENT_SOURCE_DIR}/src/kas_utils ${CMAKE_INSTALL_PREFIX}/lib/python3/dist-packages/kas_utils SYMBOLIC)"
)

# create environment hook for ROS2
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/share/${PROJECT_NAME}/hook/python_path.dsv
    "prepend-non-duplicate;PYTHONPATH;lib/python3/dist-packages"
)
install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/share/${PROJECT_NAME}/hook/python_path.dsv
    DESTINATION share/${PROJECT_NAME}/hook
)
