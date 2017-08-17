cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(
  clupp
  VERSION 0.0.1
  LANGUAGES CXX
)

# add cmake modules to the include path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(
  ${PROJECT_NAME}
  include/clupp/distance.hpp
  include/clupp/pam.hpp
  src/distance.cpp
  src/pam.cpp
)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC Eigen3::Eigen
)
