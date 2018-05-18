cmake_minimum_required(VERSION 3.10)
project(proj)

set(CMAKE_CXX_STANDARD 11)

FIND_PACKAGE(PkgConfig REQUIRED)



# gui library gtkmm-3.0
pkg_check_modules(GTKMM gtkmm-3.0)
link_directories(${GTKMM_LIBRARY_DIRS})
include_directories(include ${GTKMM_INCLUDE_DIRS})


# jpg library libjpeg
pkg_check_modules(LIBJPG libjpeg)
link_directories(${LIBJPG_LIBRARY_DIRS})
include_directories(include ${LIBJPG_INCLUDE_DIRS})


# add executable
add_executable(proj main.cpp simulation.cpp simulation.h)

target_link_libraries(proj ${GTKMM_LIBRARIES})
target_link_libraries(proj ${LIBJPG_LIBRARIES})
