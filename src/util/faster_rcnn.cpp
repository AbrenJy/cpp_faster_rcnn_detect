#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include "caffe/caffe.hpp"
#include "gpu_nms.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "faster_rcnn.hpp"
using namespace caffe;
using namespace std;

static int MAX_SIZE = 1000;
static int SCALE_SIZE = 600;
static float CONF_THRESH = 0.7;
static float NMS_THRESH = 0.3;
static int CLASS_NUM = 21;

Detector::Detector(const string& model_file, const string& weights_file,
    int class_num, int max_size, int scale_size, float conf_thresh, float nms_thresh) {
    net_ = shared_ptr<Net<float> >(new Net<float>(model_file, caffe::TEST));
    net_->CopyTrainedLayersFrom(weights_file);
    
    CLASS_NUM = class_num;
    MAX_SIZE = max_size;
    SCALE_SIZE = scale_size;
    CONF_THRESH = conf_thresh;
    NMS_THRESH = nms_thresh;

    cout << "Detector init success!" << endl;
}

vector<vector<int> > Detector::Detect(cv::Mat & cv_img)
{
    vector<vector<int> > bboxes;
    if(cv_img.empty()){
    	std::cout<<"Bad image!"<<endl;
    	return bboxes;
    }

    // minus means
    cv::Mat cv_new(cv_img.rows, cv_img.cols, CV_32FC3, cv::Scalar(0,0,0));
	for (int h = 0; h < cv_img.rows; ++h ){
		for (int w = 0; w < cv_img.cols; ++w){
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[0])-float(102.9801);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[1])-float(115.9465);
			cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_img.at<cv::Vec3b>(cv::Point(w, h))[2])-float(122.7717);
		}
	}

    // resize: scale image to (1000, y) or (x, 600)
    const int  MAX_SIZE = 1000;
    const int  SCALE_SIZE = 600;
	int max_side = max(cv_img.rows, cv_img.cols);
	int min_side = min(cv_img.rows, cv_img.cols);
    cout << "(width, height) = (" << cv_img.cols << ", " << cv_img.rows << ")" << endl;
    float img_scale = float(SCALE_SIZE) / float(min_side);
    if (round(float(max_side) * img_scale) > MAX_SIZE) {
        img_scale = float(MAX_SIZE) / float(max_side);
    }
    cout << "img_scale: " << img_scale << endl;

	int height = int(cv_img.rows * img_scale);
	int width = int(cv_img.cols * img_scale);
    cout << "re-scaled (width, height) = (" << width << ", " << height << ")" << endl;
	cv::Mat cv_resized;
	cv::resize(cv_new, cv_resized, cv::Size(width, height));

	float data_buf[height*width*3];
	float *boxes = NULL;
	float *pred = NULL;
	float *pred_per_class = NULL;
	float *sorted_pred_cls = NULL;
	int *keep = NULL;
	const float* bbox_delt;
	const float* rois;
	const float* scores;
	int num_out;
	int num;

    /* 从(高,宽,3)变换为(3, 高, 宽), data_buf的内存布局为
    B, G, R 每个区域的高度是height, 所以每个区域的起始高度如下:
        -------------   -> h
        |           |
        |    B      |
        |           |
        -------------   -> height + h
        |           |
        |    G      |
        |           |
        -------------   -> 2 * height + h
        |           |
        |    R      |
        |           |
        -------------
    */
	for (int h = 0; h < height; ++h) {
		for (int w = 0; w < width; ++w) {
			data_buf[(0 * height + h) * width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);// Blue
			data_buf[(1 * height + h) * width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);// Green
			data_buf[(2 * height + h) * width + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);// Red
		}
	}

    Blob<float> *input_layer1 = net_->input_blobs()[0];
    Blob<float> *input_layer2 = net_->blob_by_name("data").get();
    cout << "input_layer1: " << input_layer1 << ", input_layer2: " << input_layer2 << endl;
 
	net_->blob_by_name("data")->Reshape(1, 3, height, width);
    net_->Reshape();
	Blob<float> * input_blobs= net_->input_blobs()[0];
    cout << "input_blobs->count(): " << input_blobs->count() << endl;
    switch(Caffe::mode()){
    case Caffe::CPU:
        memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
        break;
    case Caffe::GPU:
        caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), data_buf, input_blobs->mutable_gpu_data());
        break;
    default:
        LOG(FATAL)<<"Unknow Caffe mode";
    }

	float im_info[3];
	im_info[0] = cv_resized.rows;
	im_info[1] = cv_resized.cols;
	im_info[2] = img_scale;

	net_->blob_by_name("im_info")->set_cpu_data(im_info);
	net_->ForwardFrom(0);
	bbox_delt = net_->blob_by_name("bbox_pred")->cpu_data();
	num = net_->blob_by_name("rois")->num();
    cout << "net_->blob_by_name(\"rois\")->num(): " << num << endl;

    // ROIs
	rois = net_->blob_by_name("rois")->cpu_data();
    // scores <- cls_prob
	scores = net_->blob_by_name("cls_prob")->cpu_data();

	boxes = new float[num*4];
	pred = new float[num*5*CLASS_NUM];
	pred_per_class = new float[num*5];
	sorted_pred_cls = new float[num*5];
	keep = new int[num];

	for (int n = 0; n < num; n++){
		for (int c = 0; c < 4; c++){
            /*rois一行是5个数据,所以是n*5. 每行第一个数据为0, 所以取1~4位置的数据.
            boxes一行是4个数据,所以是n*4.
            rois记录的是缩放后的大小, 除以img_scale转换为原图中的大小.*/
			boxes[n * 4 + c] = rois[n * 5 + c + 1] / img_scale;
		}
	}

	bbox_transform_inv(num, bbox_delt, scores, boxes, pred, cv_img.rows, cv_img.cols);
 
    for (int i = 1; i < CLASS_NUM; i++) {
	    for (int j = 0; j< num; j++){
            for (int k=0; k<5; k++){
	    	    pred_per_class[j*5+k] = pred[(i*num+j)*5+k];
	    	}
	    }
	    boxes_sort(num, pred_per_class, sorted_pred_cls);
	    _nms(keep, &num_out, sorted_pred_cls, num, 5, NMS_THRESH, 0);
        cout << "num: " << num << ", num_out: " << num_out << endl;
	    for(int i_ = 0;sorted_pred_cls[keep[i_]*5+4] > CONF_THRESH && i_ < num_out;++i_){
            cout << "got " << i_ << endl;
            vector<int> bbox;
	        bbox.push_back((int)sorted_pred_cls[keep[i_]*5+0]);
	    	bbox.push_back((int)sorted_pred_cls[keep[i_]*5+1]);	
	    	bbox.push_back((int)sorted_pred_cls[keep[i_]*5+2] - (int)sorted_pred_cls[keep[i_]*5+0]);
	    	bbox.push_back((int)sorted_pred_cls[keep[i_]*5+3] - (int)sorted_pred_cls[keep[i_]*5+1]);	
	    	bboxes.push_back(bbox);
	    }
    }


	delete []boxes;
	delete []pred;
	delete []pred_per_class;
	delete []keep;
	delete []sorted_pred_cls;

	return bboxes;

}

void Detector::boxes_sort(const int num, const float* pred, float* sorted_pred)
{
	vector<Info> my;
	Info tmp;
	for (int i = 0; i < num; i++){
		tmp.score = pred[i * 5 + 4];
		tmp.head = pred + i * 5;
		my.push_back(tmp);
	}
	std::sort(my.begin(), my.end(), compare);
	for (int i = 0; i < num; i++){
		for (int j = 0; j < 5; j++){
			sorted_pred[i*5+j] = my[i].head[j];
		}
    }
}

void Detector::bbox_transform_inv(int num, const float* box_deltas, const float* scores,
    float* boxes, float* pred, int img_height, int img_width)
{
	float width, height, ctr_x, ctr_y, dx, dy, dw, dh, pred_ctr_x, pred_ctr_y, pred_w, pred_h;

	for (int i = 0; i < num; i++) {
        //boxes大小是num*4, 一行4个数据,分别是(xmin, ymin, xmax, ymax), x轴向右,y轴向下
		width = boxes[i * 4 + 2] - boxes[i * 4 + 0] + 1.0;// xmax - xmin + 1
		height = boxes[i * 4 + 3] - boxes[i * 4 + 1] + 1.0;// ymax - ymin + 1
		ctr_x = boxes[i * 4 + 0] + 0.5 * width;
		ctr_y = boxes[i * 4 + 1] + 0.5 * height;

		for (int j = 0; j < CLASS_NUM; j++) {
            //box_deltas 的大小是num * CLASS_NUM * 4, 一共num行,每行CLASS_NUM * 4个数据
            // dx是每行中第0, 4, 8...位置的数据, dy是1, 5, 9...位置的数据
            // dw是每行中第2, 6, 10...位置的数据, dh是3, 7, 11...位置的数据
			dx = box_deltas[(i * CLASS_NUM + j) * 4 + 0];// i * CLASS_NUM * 4 + j * 4 表示第i行中4*j的位置
			dy = box_deltas[(i * CLASS_NUM + j) * 4 + 1];
			dw = box_deltas[(i * CLASS_NUM + j) * 4 + 2];
			dh = box_deltas[(i * CLASS_NUM + j) * 4 + 3];

			pred_ctr_x = ctr_x + width * dx;
			pred_ctr_y = ctr_y + height * dy;
			pred_w = width * exp(dw);
			pred_h = height * exp(dh);

            // pred的大小是num*5*CLASS_NUM, 
			pred[(j * num + i) * 5 + 0] = max(min(pred_ctr_x - 0.5 * pred_w, img_width - 1), 0);
			pred[(j * num + i) * 5 + 1] = max(min(pred_ctr_y - 0.5 * pred_h, img_height - 1), 0);
			pred[(j * num + i) * 5 + 2] = max(min(pred_ctr_x + 0.5 * pred_w, img_width - 1), 0);
			pred[(j * num + i) * 5 + 3] = max(min(pred_ctr_y + 0.5 * pred_h, img_height - 1), 0);
            // scores的大小是(num * CLASS_NUM), i*CLASS_NUM+j表示每一行中每一项的分数
			pred[(j * num + i) * 5 + 4] = scores[i * CLASS_NUM + j];
		}
   }
}
