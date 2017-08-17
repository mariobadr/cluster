cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(
  clupp
  VERSION 0.0.1
  LANGUAGES CXX
)

add_library(
  ${PROJECT_NAME}
  include/clupp/cluster.hpp
  src/cluster.cpp
)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>"
)
