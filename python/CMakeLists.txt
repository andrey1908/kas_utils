cmake_minimum_required(VERSION 3.16)
project(kas_utils_python)

find_package(kas_utils_cpp REQUIRED)

file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/share/${PROJECT_NAME}/hook/python_path.dsv
    "prepend-non-duplicate;PYTHONPATH;lib/python3/dist-packages")

install(
    FILES package.xml
    DESTINATION share/${PROJECT_NAME}
)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/share/${PROJECT_NAME}/hook/python_path.dsv
    DESTINATION share/${PROJECT_NAME}/hook)

install(
    DIRECTORY ${CMAKE_SOURCE_DIR}/src/kas_utils
    DESTINATION lib/python3/dist-packages
    PATTERN "*.pyc" EXCLUDE
    PATTERN "__pycache__" EXCLUDE
)

install(
    FILES ${kas_utils_cpp_PYTHON_LIBRARIES}
    DESTINATION lib/python3/dist-packages/kas_utils
)
