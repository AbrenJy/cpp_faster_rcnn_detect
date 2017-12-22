set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_C_COMPILER g++)

set(
	LIB_SRC_LIST
	faster_rcnn.cpp
	)

include_directories(
	${PROJECT_SOURCE_DIR}/include
	${faster_rcnn_path}/caffe-fast-rcnn/include
	${faster_rcnn_path}/caffe-fast-rcnn/include/caffe
	${faster_rcnn_path}/lib/nms
	/usr/include
	/usr/local/include
	/usr/local/cuda/include
	)

add_library(faster_rcnn SHARED ${LIB_SRC_LIST})
