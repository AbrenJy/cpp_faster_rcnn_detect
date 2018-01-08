set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

set(
    SRC_LIST
    faster_rcnn_detect.cpp
    )

set(CMAKE_CXX_FLAGS "-g -std=c++11")

#same as set(MY_PROJECT_DIR ${PROJECT_SOURCE_DIR})
set(MY_PROJECT_DIR ${faster_rcnn_cpp_SOURCE_DIR})
message("\${MY_PROJECT_DIR} :" ${MY_PROJECT_DIR})

set(CMAKE_INSTALL_RPATH ${MY_PROJECT_DIR}/lib)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH true)

find_package( OpenCV REQUIRED )

include_directories(
	${MY_PROJECT_DIR}/include
	${faster_rcnn_path}/lib/nms
	${faster_rcnn_path}/caffe-fast-rcnn/include
	${MY_PROJECT_DIR}/lib
	/usr/local/include
	/usr/include
	/usr/local/cuda/include
	)

link_directories(
	${MY_PROJECT_DIR}/lib
	${faster_rcnn_path}/caffe-fast-rcnn/build/lib
	${faster_rcnn_path}/lib/nms
	/usr/local/lib
	/usr/lib
	)

add_executable(
	faster_rcnn_detect
	${SRC_LIST}
	)

add_dependencies(faster_rcnn_detect faster_rcnn)

add_custom_command(TARGET faster_rcnn_detect PRE_BUILD
    COMMAND cmake -E copy ${faster_rcnn_path}/lib/nms/gpu_nms.so
    ${MY_PROJECT_DIR}/lib/libgpu_nms.so)

target_link_libraries(
	faster_rcnn_detect
	libfaster_rcnn.so
	libcaffe.so
	libgpu_nms.so
    ${OpenCV_LIBS}
	libboost_system.so
    libyaml-cpp.so
    /usr/lib/x86_64-linux-gnu/libgflags.so
	)

install(TARGETS faster_rcnn_detect
    DESTINATION $ENV{HOME}/bin)

