cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(
  pam-1D
  VERSION 0.0.1
  LANGUAGES CXX
)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(
  ${PROJECT_NAME}
  PUBLIC cluster
)

set_target_properties(
  ${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
)