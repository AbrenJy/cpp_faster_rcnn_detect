**Note:**
This repo is forked from https://github.com/QiangXie/libfaster_rcnn_cpp .
But I found another similar code: https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus
Most of code is same.
And there is blog to explain the code: http://blog.csdn.net/xyy19920105/article/details/50440957

**What I have done:**
1. delete redundant 'include' and undependent .so files
2. set `faster_rcnn_path` for all CMakeLists.txt. You only need to set only one viriable (`faster_rcnn_path`) to configure env.
3. You don't have to copy gpu_nms.so manually. This is done by cmake now.

This project is Faster-rcnn detector C++ version demo, if you want to learn more about Faster-rcnn, please click [py-faster-rcnn][1].

**0 Precondition**

Make sure your `py-faster-rcnn` work well.

**1 Clone the project repository**

```
    git clone https://github.com/galian123/libfaster_rcnn_cpp
```

**2 set `faster_rcnn_path` in `src/CMakeLists.txt`**

Like this : `set(faster_rcnn_path "~/git/py-faster-rcnn/")`

If you forget to set `faster_rcnn_path`, error will happen if you run `cmake ..`.

```
CMake Error at src/CMakeLists.txt:4 (message):
  Error: please set ${faster_rcnn_path} before run cmake.

    ${faster_rcnn_path} is where your 'py-faster-rcnn' exists.

-- Configuring incomplete, errors occurred!
```    

**3 Modify `main.cpp`**

Set your path of `test.prototxt` and `.caffemodel` file in `main.cpp`.

**4 Build**

Current folder is `libfaster_rcnn_cpp`.

```
    mkdir build
    cd build
    cmake ..
    make
```
 
**5 Run the program**

Current folder is `libfaster_rcnn_cpp`.

**NOTE: run `main` in `./bin` folder. Because hardcode 'test1.jpg' is in `bin` folder.**

```
    cd bin
    ./main
```

This program will detect test1.jpg in bin folder, and print the detected vehicle bounding box, then rectangle bounding box and saved as test.jpg. If you need modify this project to do more, see main.cpp.

**6 TODO**

Make main(main.cpp) can receive arguments, or save settings to yaml file.

**7 Fix protobuf version error**

```
I1222 20:17:27.105358 24948 layer_factory.hpp:77] Creating layer proposal
[libprotobuf FATAL google/protobuf/stubs/common.cc:61] This program requires version 3.4.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1.  Please update your library.  If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library.  (Version verification failed in "google/protobuf/descriptor.pb.cc".)
terminate called after throwing an instance of 'google::protobuf::FatalException'
  what():  This program requires version 3.4.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1.  Please update your library.  If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library.  (Version verification failed in "google/protobuf/descriptor.pb.cc".)
Aborted (core dumped)
```

**Solution:**
Run `pip show protobuf` to show protobuf version in python (installed via pip).
Run `protoc --version` to show protobuf version (installed via apt-get).
These two version are not same.

Uninstall protobuf: `sudo pip uninstall protobuf`
Install protobuf with version 2.6.1: `sudo pip install protobuf==2.6.1`
Then recompile `py-faster-rcnn'.

Refer to https://github.com/BVLC/caffe/issues/5711


[1]: https://github.com/rbgirshick/py-faster-rcnn "py-faster-rcnn"


