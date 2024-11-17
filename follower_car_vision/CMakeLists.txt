cmake_minimum_required(VERSION 3.8)
project(follower_car_vision)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

set(DEPENDENCIES
  "rclpy"
  "sensor_msgs"
  "geometry_msgs"
  "cv_bridge"
)

install(TARGETS ${EXEC_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  src/follower_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(rclpy)
ament_export_dependencies(sensor_msgs)
ament_export_dependencies(geometry_msgs)
ament_export_dependencies(cv_bridge)
ament_package()
