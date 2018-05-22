

cmake_minimum_required(VERSION 2.8) # cmake verson，指定cmake版本 
add_definitions(-std=c++11 -g)

project (ssd_detected)

file(GLOB_RECURSE ssd_detected_SOURCES "src/*.cpp")

set (ssd_detected_INCLUDE_DIRS "")
foreach (_headerFile ${ssd_detected_HEADERS})
    get_filename_component(_dir ${_headerFile} PATH)
    list (APPEND ssd_detected_INCLUDE_DIRS ${_dir})
endforeach()


set (caffe_ssd_PATH  /home/gjw/caffe-ssd)

set (caffe_ssd_INCLUDE_DIRS ${caffe_ssd_PATH}/include)
set (caffe_ssd_BUILD_DIRS ${caffe_ssd_PATH}/.build_release/src)

set (cuda_INCLUDE_DIRS /usr/local/cuda-8.0/include)
set (boost_system_INCLUDE_DIRS /usr/include/boost)
set (gflags_INCLUDE_DIRS /usr/local/include/gflags)
set (glog_INCLUDE_DIRS /usr/include/glog)
set (gtest_INCLUDE_DIRS /usr/local/include/gtest)



include_directories(
	${ssd_detected_INCLUDE_DIRS} 
	${caffe_ssd_INCLUDE_DIRS} 
	${caffe_ssd_BUILD_DIRS} 
	${cuda_INCLUDE_DIRS} 
	${gflags_INCLUDE_DIRS}  
	${gtest_INCLUDE_DIRS} 
	${glog_INCLUDE_DIRS} 
	${boost_system_INCLUDE_DIRS}
)

### OpenCV的配置 ###
set (OpenCV_DIR /usr)  
find_package(OpenCV 2 REQUIRED)

add_executable (ssd_detected ${ssd_detected_SOURCES})

set (caffe_ssd_LIBS ${caffe_ssd_PATH}/build/lib/libcaffe.so)
set (boost_system_LIBS /usr/lib/x86_64-linux-gnu/libboost_system.so)
set (gflags_LIBS /usr/lib/x86_64-linux-gnu/libgflags.so)
set (glog_LIBS /usr/local/lib/libglog.so)


target_link_libraries( ssd_detected 
	${OpenCV_LIBS} 
	${caffe_ssd_LIBS} 
	${gflags_LIBS} 
	${gtest_LIBS} 
	${glog_LIBS} 
	${boost_system_LIBS}
)


