set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

set(
	SRC_LIST
	main.cpp
	)

find_package( OpenCV REQUIRED )

include_directories(
	${PROJECT_SOURCE_DIR}/include
	${faster_rcnn_path}/lib/nms
	${faster_rcnn_path}/caffe-fast-rcnn/include
	${PROJECT_SOURCE_DIR}/lib
	/usr/local/include
	/usr/include
	/usr/local/cuda/include
	)

link_directories(
	${PROJECT_SOURCE_DIR}/lib
	${faster_rcnn_path}/caffe-fast-rcnn/build/lib
	${faster_rcnn_path}/lib/nms
	/usr/local/lib
	/usr/lib
	)

add_executable(
	main
	${SRC_LIST}
	)

add_dependencies(main faster_rcnn)

add_custom_command(TARGET main PRE_BUILD
    COMMAND cmake -E copy ${faster_rcnn_path}/lib/nms/gpu_nms.so
    ${PROJECT_SOURCE_DIR}/lib/libgpu_nms.so)

target_link_libraries(
	main
	libfaster_rcnn.so
	libcaffe.so
	libgpu_nms.so
	${OpenCV_LIBS}
	libboost_system.so
	)
