cmake_minimum_required( VERSION 2.8 )
project( g2o_ba_example )

set( CMAKE_BUILD_TYPE "Release" )
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
set( G2O_ROOT ${PROJECT_SOURCE_DIR}/g2o ) 
find_package( G2O )

include_directories( ${G2O_INCLUDE_DIRS} )
include_directories( ${EIGEN3_INCLUDE_DIR})
add_executable( BA main.cpp )
target_link_libraries( BA 
    ${OpenCV_LIBS}
    g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sba
    )