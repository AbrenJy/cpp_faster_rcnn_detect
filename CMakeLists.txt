project (faster_rcnn_cpp)
cmake_minimum_required(VERSION 2.8)

# set(faster_rcnn_path "~/git/py-faster-rcnn/")
if(NOT DEFINED faster_rcnn_path)
    message(FATAL_ERROR "Error: please set \${faster_rcnn_path} before run cmake." 
        "\n\${faster_rcnn_path} is where your 'py-faster-rcnn' exists.")
endif(NOT DEFINED faster_rcnn_path)        

add_subdirectory(src/util)
add_subdirectory(src/main)
