cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(
  cluster
  VERSION 0.0.1
  LANGUAGES CXX
)

# include potential build options
include(options.cmake)

# add cmake modules to the include path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

add_library(
  ${PROJECT_NAME}
  include/cluster/distance.hpp
  include/cluster/pam.hpp
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

if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX "-debug")
endif()

set_target_properties(
  ${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX}
)

if(CLUSTER_BUILD_EXAMPLES)
  message(STATUS "cluster: Build examples option enabled.")
  add_subdirectory(examples)
endif()

# create cmake package
set(CLUSTER_PACKAGE_DESTINATION "lib/cmake/${PROJECT_NAME}")
include(CMakePackageConfigHelpers)

# files to be generated
set(CLUSTER_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(CLUSTER_VERSION_CONFIG "${CLUSTER_GENERATED_DIR}/${PROJECT_NAME}-config-version.cmake")
set(CLUSTER_PROJECT_CONFIG "${CLUSTER_GENERATED_DIR}/${PROJECT_NAME}-config.cmake")
set(CLUSTER_EXPORTED_TARGET "${PROJECT_NAME}-target")

# use the version defined by the project command we issued at the top
write_basic_package_version_file(
  "${CLUSTER_VERSION_CONFIG}" COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
  "cmake/ClusterConfig.cmake.in"
  "${CLUSTER_PROJECT_CONFIG}"
  INSTALL_DESTINATION "${CLUSTER_PACKAGE_DESTINATION}"
)

# install the library binary and export its properties to a .cmake file
install(
  TARGETS ${PROJECT_NAME}
  EXPORT "${CLUSTER_EXPORTED_TARGET}" # CMakeFiles/Export/
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# install the library headers
install(
  DIRECTORY include/cluster
  DESTINATION include
)

# install the cluster-config.cmake file and its version info
install(
  FILES "${CLUSTER_PROJECT_CONFIG}" "${CLUSTER_VERSION_CONFIG}"
  DESTINATION "${CLUSTER_PACKAGE_DESTINATION}"
)

# install the cluster-target* information
install(
  EXPORT "${CLUSTER_EXPORTED_TARGET}"
  NAMESPACE "cluster::"
  DESTINATION "${CLUSTER_PACKAGE_DESTINATION}"
)