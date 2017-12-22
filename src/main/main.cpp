#include "faster_rcnn.hpp"
using namespace cv;
int main()
{
    string py_faster_rcnn_path = "/home/<your_name>/git/py-faster-rcnn/"; // NOTE: set your path
    string net_type = "VGG_CNN_M_1024";
    string model_file = py_faster_rcnn_path + "models/pascal_voc/" + net_type + "/faster_rcnn_end2end/test.prototxt";
    string weights_file = py_faster_rcnn_path + "output/faster_rcnn_end2end/voc_2007_trainval/vgg_cnn_m_1024_faster_rcnn_iter_70000.caffemodel";
    int GPUID=0;
    vector<vector<int> > ans;
    Caffe::SetDevice(GPUID);
    Caffe::set_mode(Caffe::GPU);
    Detector det = Detector(model_file, weights_file);
    cv::Mat im = cv::imread("test1.jpg");
    ans = det.Detect(im);
    for(int i = 0;i < ans.size();++i){
        for(int j = 0;j < ans[i].size();j++){
            cout << ans[i][j] << " ";
        }
	rectangle(im,cvPoint(ans[i][0],ans[i][1]),cvPoint(ans[i][2] + ans[i][0],ans[i][3] + ans[i][1]),Scalar(0,0,255),1,1,0);
        cout << endl;
    }
    imwrite("test.jpg",im);
    return 0;
}
