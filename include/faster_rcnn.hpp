#ifndef FASTER_RCNN_HPP
#define FASTER_RCNN_HPP
#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <math.h>
#include <fstream>
#include "caffe/caffe.hpp"
#include "gpu_nms.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace caffe;
using namespace std;

#define max(a, b) (((a)>(b)) ? (a):(b))
#define min(a, b) (((a)<(b)) ? (a):(b))


class Detector {
    public:
        Detector(const string& model_file, const string& weights_file,
            int class_num, int max_size, int scale_size, float conf_thresh, float nms_thresh);
        vector<vector<int> > Detect(cv::Mat & cv_img);
        void bbox_transform_inv(const int num, const float* box_deltas, const float* pred_cls, float* boxes, float* pred, int img_height, int img_width);
        void boxes_sort(int num, const float* pred, float* sorted_pred);

    private:
        boost::shared_ptr<Net<float> > net_;
        Detector(){}
};

//Using for box sort
struct Info{
    float score;
    const float* head;
};
bool compare(const Info& Info1, const Info& Info2) {
    return Info1.score > Info2.score;
}
#endif
