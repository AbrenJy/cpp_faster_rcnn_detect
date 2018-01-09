**Note:**
This repo is forked from https://github.com/QiangXie/libfaster_rcnn_cpp . <br>
But I found another similar code: https://github.com/YihangLou/FasterRCNN-Encapsulation-Cplusplus <br>
Most of code is same.<br>
And there is blog to explain the code: http://blog.csdn.net/xyy19920105/article/details/50440957

**What I have done:**
updated on 2018-01-06.
1. Delete redundant 'include' and undependent .so files
2. Set `faster_rcnn_path` for all CMakeLists.txt. 
You only need to set only one viarable (`faster_rcnn_path`) to configure environment.
3. You don't have to copy gpu_nms.so manually. This is done by cmake automatically now.
4. Put some configs into yml file in 'config' folder. Most of the options can be set in yml file.
5. Use gflags to process parameters. The config file (yml file) follows '-yml_file'. 
You can also set '<KEY> <VALUE>' to override the same value in yml file. Run `faster_rcnn_detect -helpshort` for more help info.
6. Fix some issues: 
(1) fix the image scale rate. Big image can scale to small. Small image can scale to big.
(2) check all object classes (e.g. CLASS_NUM is 21).

This project is Faster-rcnn detector C++ version, the code flow is almost same as demo.py.
If you want to learn more about Faster-rcnn, please click [py-faster-rcnn][1].

## **0. Precondition**

Make sure your `py-faster-rcnn` works well.

## **1. Download code**

```
git clone https://github.com/galian123/cpp_faster_rcnn_detect
```

## **2 set `faster_rcnn_path` in `CMakeLists.txt`**

Like this : `set(faster_rcnn_path "~/git/py-faster-rcnn/")`

If you forget to set `faster_rcnn_path`, error will happen if you run `cmake ..`.

```
CMake Error at src/CMakeLists.txt:4 (message):
  Error: please set ${faster_rcnn_path} before run cmake.
  ${faster_rcnn_path} is where your 'py-faster-rcnn' exists.
-- Configuring incomplete, errors occurred!
```    

## **3. Install dependent libs**

* install gflags:

`sudo apt-get install -y --no-install-recommends libgflags-dev`

* install yaml lib: 

```
git clone https://github.com/jbeder/yaml-cpp
cd yaml-cpp
mkdir build
cd build
cmake ..
make -j8
sudo make install
```

## **4 Modify yml file in `config` folder.**

Set correct MODEL_FILE (test.prototxt) and TRAINED_FILE (xxx.caffemodel).

## **5 Build**

Current folder is `cpp_faster_rcnn_detect`.

```
mkdir build
cd build
cmake ..
make
```
 
## **5 Run**

Current folder is `cpp_faster_rcnn_detect/build`.

```
./faster_rcnn_detect -imgdir ../tested_images -yml_file ../config/faster_rcnn_end2end.yml
```

Default output folder is `./labeled_images`. You can find the results of 3 images for example.

To get more help, run `./faster_rcnn_detect -helpshort`.

You can pass parameters override the value in yml file, like this: <br>
`./faster_rcnn_detect -imgdir ../tested_images -yml_file ../config/faster_rcnn_end2end.yml GPUID 1 CONF_THRESH 0.6` <br>
`GPUID 1 CONF_THRESH 0.6` are `<KEY> <VALUE>` pairs. The `<KEY>` can be the key in yml file.

To display class name of the object and score:
```
./faster_rcnn_detect -showlabel -imgdir ../tested_images -yml_file ../config/faster_rcnn_end2end.yml
```
`-showlabel`: default value is true, so you don't need to set it.

To display rectangle only: use `-noshowlabel` or `-showlabel=false`
``
./faster_rcnn_detect -imgdir ../tested_images/ -yml_file ../config/faster_rcnn_end2end.yml -outdir ./labeled_images_simple -noshowlabel
```

## **7 Fix protobuf version error**

```
I1222 20:17:27.105358 24948 layer_factory.hpp:77] Creating layer proposal
[libprotobuf FATAL google/protobuf/stubs/common.cc:61] This program requires version 3.4.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1.  Please update your library.  If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library.  (Version verification failed in "google/protobuf/descriptor.pb.cc".)
terminate called after throwing an instance of 'google::protobuf::FatalException'
  what():  This program requires version 3.4.0 of the Protocol Buffer runtime library, but the installed version is 2.6.1.  Please update your library.  If you compiled the program yourself, make sure that your headers are from the same version of Protocol Buffers as your link-time library.  (Version verification failed in "google/protobuf/descriptor.pb.cc".)
Aborted (core dumped)
```

**Solution:**

Run `pip show protobuf` to show protobuf version in python (installed via pip). <br>
Run `protoc --version` to show protobuf version (installed via apt-get). <br>
These two version are not same.

Uninstall protobuf: `sudo pip uninstall protobuf` <br>
Install protobuf with version 2.6.1: `sudo pip install protobuf==2.6.1` <br>
Then recompile `py-faster-rcnn'.

Refer to https://github.com/BVLC/caffe/issues/5711


[1]: https://github.com/rbgirshick/py-faster-rcnn "py-faster-rcnn"


