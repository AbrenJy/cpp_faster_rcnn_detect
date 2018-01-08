/*
Precondition: 
Make sure your py-faster-rcnn works well.

Modifications:
1. Put the const strings into yaml file.
Use yaml lib, so you have to 
(1) Install cmake: "sudo apt-get install cmake" (Maybe you have installed yet)
(2) Download yaml source file: "git clone https://github.com/jbeder/yaml-cpp"
(3) run "cmake ..", "make -j8", "sudo make install" to install yaml-cpp

2. Use gflags to allow user to pass parameters
(1) Install gflags: "sudo apt-get install -y --no-install-recommends libgflags-dev"

Modified by galian.
2018.01
*/
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <dirent.h>
#include <assert.h>
#include "faster_rcnn.hpp"
#include "gflags/gflags.h"
#include "yaml-cpp/yaml.h"

using namespace cv;

const char *KEY_GPUID = "GPUID";
const char *KEY_CLASS_NUM = "CLASS_NUM";
const char *KEY_MAX_SIZE = "MAX_SIZE";
const char *KEY_SCALE_SIZE = "SCALE_SIZE";
const char *KEY_CONF_THRESH = "CONF_THRESH";
const char *KEY_NMS_THRESH = "NMS_THRESH";
const char *KEY_MODEL_FILE = "MODEL_FILE";
const char *KEY_TRAINED_FILE = "TRAINED_FILE";

DEFINE_string(imgdir, "", "Set the input directory contains images.");
DEFINE_string(outdir, "./labeled_images",
    "Set the output directory which saves labeled images.");
DEFINE_string(yml_file, "", "Set config file (xxx.yml)");
DEFINE_bool(showlabel, true, "show class name of the object and score.");
DEFINE_bool(verbose, false, "show more logs");

static bool verbose = false;
static bool showlabel = true;

bool checkDirExist(const char *dirName) {
    struct stat dirInfo = {0};
    int statRet = stat(dirName, &dirInfo);
    if (statRet == -1 && errno == ENOENT) {
        cout << dirName << " not exist" << endl;
        return false;
    } else {
        if ((dirInfo.st_mode & S_IFDIR) != S_IFDIR) {
            cout << dirName << " is not directory." << endl;
            return false;
        }
    }
    return true;
}

bool checkFileExist(const char *name) {
    struct stat info = {0};
    int statRet = stat(name, &info);
    if (statRet == -1 && errno == ENOENT) {
        cout << name <<" not exist" << endl;
        return false; 
    } else {
        if (!S_ISREG(info.st_mode)) {
            cout << "Error: " << name << " is not regular file." << endl;
            return false; 
        }
    }   
    return true;
}

bool tryMakedir(const char *dirName) {
    if (!checkDirExist(dirName)) {
        int ret = mkdir(dirName, 0775);
        if (ret != 0) {
            cout << "Error: mkdir " << dirName << " failed, errno: " << errno << ", " << strerror(errno) << endl;
            return false;
        }
        cout << "Info: mkdir " << dirName << " success" << endl;
    } else {
        cout << "Info: " << dirName << " is OK." << endl;
    }   
    return true;
}

int filterImg(const struct dirent *entry) {
    if (entry != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            return 0;
        }
        char *p = strrchr((char*)entry->d_name, '.');
        if (p != NULL &&
            (strcasecmp(p, ".jpg") == 0 || strcasecmp(p, ".png") == 0
            || strcasecmp(p, ".jpeg") == 0)) {
            return 1;
        }
    }
    return 0;
}

void tryAddSlash(char dir[]) {
    int len = 0;
    len = strlen(dir);
    if (dir[len-1] != '/') {
        dir[len] = '/';
        dir[len+1] = 0;
    }
}

void drawRectOnImage(cv::Mat& im, vector<float> pos) {
    int thickness = 2;
    Scalar color(255, 0, 0);// color is blue.

    Mat overlay;
    im.copyTo(overlay);
	rectangle(overlay, cvPoint(pos[0], pos[1]),
        cvPoint(pos[2] + pos[0], pos[3] + pos[1]),
        color, thickness);

    double alpha = 0.5;
    cv::addWeighted(overlay, alpha, im, 1 - alpha, 0, im);
}

// voffset: vertical offset
int drawTextOnImage(cv::Mat& im, vector<float> pos, float voffset, string text) {
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1.0;
    int thickness = 1;

    int baseline = 0;
    Size textSize = getTextSize(text, fontFace, fontScale, thickness, &baseline);

    Mat overlay;
    im.copyTo(overlay);

    rectangle(overlay, cvPoint(pos[0], pos[1] + voffset),
        cvPoint(pos[0] + textSize.width, pos[1] + textSize.height + thickness + voffset),
        Scalar(255, 128, 128), CV_FILLED);

    Point textOrg(pos[0], pos[1] + textSize.height + thickness + voffset);
    cv::putText(overlay, text, textOrg,
        fontFace, fontScale, Scalar(255, 255, 255), thickness, LINE_AA);
    double alpha = 0.75;
    cv::addWeighted(overlay, alpha, im, 1 - alpha, 0, im);
    return textSize.height + thickness;
}

int main(int argc, char **argv) {
    gflags::SetUsageMessage("faster_rcnn_detect is used to identify objects in image file.\n"
        "Please use 'faster_rcnn_detect -helpshort' to get only help message for faster_rcnn_detect. \n"
        "Usage:\n"
        "    faster_rcnn_detect -imgdir <input_image_folder> [OPTIONS] [ARGUMENTS]\n"
        "    faster_rcnn_detect -imgdir <input_image_folder> -outdir <output_labeled_images_folder> [OPTIONS] [ARGUMENTS]\n"
        "You can set the options in './config/faster_rcnn_xxx.yml' and pass the yml file to '-yml_file'.\n"
        "Arguments from command line will override the config in yml file.\n"
        "NOTE: if you don't set '-yml_file', you must set all arguments needed.\n"
        "You'd better set '-yml_file', and add some arguments if you want to override the value in yml file.\n"
        "[ARGUMENTS] like this: <key> <value> <key> <value> ...\n"
        "    GPUID <gpuid> MODEL_FILE <model_file_path> TRAINED_FILE <trained_file> ...\n"
        "Refer to yml file to get what <key> can be.\n");
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    gflags::CommandLineFlagInfo info;
    bool ret = gflags::GetCommandLineFlagInfo("yml_file", &info);
    if (info.is_default) {
        cout << "Warning: if you don't set '-yml_file', you must set all arguments" << endl;
    }

    verbose = FLAGS_verbose;
    showlabel = FLAGS_showlabel;
    string model_file, weights_file;
    int gpuid = 0;
    string imgdir = FLAGS_imgdir;
    string outdir = FLAGS_outdir;
    int class_num, max_size, scale_size;
    float conf_thresh, nms_thresh;
    vector<string> class_names;

    const string yml_file = FLAGS_yml_file;
	if (!yml_file.empty()) {
        if (!checkFileExist(yml_file.c_str())) {
            return -1;
        }
		cout << "load config file: " << yml_file << endl;
    	YAML::Node config = YAML::LoadFile(yml_file.c_str());
        gpuid = config[KEY_GPUID].as<int>();
        model_file = config[KEY_MODEL_FILE].as<string>();
        weights_file = config[KEY_TRAINED_FILE].as<string>();

        class_num = config[KEY_CLASS_NUM].as<int>();
        conf_thresh = config[KEY_CONF_THRESH].as<float>();
        nms_thresh = config[KEY_NMS_THRESH].as<float>();
        max_size = config[KEY_MAX_SIZE].as<int>();
        scale_size = config[KEY_SCALE_SIZE].as<int>();

        const YAML::Node& class_name_node = config["CLASS_NAME"];
        for (int i = 0; i < class_name_node.size(); i++) {
            const string name = class_name_node[i].as<string>();
            if (verbose) cout << name << " ";
            class_names.push_back(name);
        }
        if (verbose) cout << endl;
	}

    // key value pairs
    assert(((argc-1) % 2) == 0);// first arg is program name
    for (int i = 1; i < argc; i += 2) {// start from 1
        char *key = argv[i];
        char *val = argv[i+1];
        cout << "key: " << key << ", value: " << val << endl;

        if (strcmp(KEY_GPUID, key) == 0) {
            gpuid = stoi(val);
        } else if (strcmp(KEY_CLASS_NUM, key) == 0) {
            class_num = stoi(val);
        } else if (strcmp(KEY_MAX_SIZE, key) == 0) {
            max_size = stoi(val);
        } else if (strcmp(KEY_SCALE_SIZE, key) == 0) {
            scale_size = stoi(val);
        } else if (strcmp(KEY_CONF_THRESH, key) == 0) {
            conf_thresh = stof(val);
        } else if (strcmp(KEY_NMS_THRESH, key) == 0) {
            nms_thresh = stof(val);
        } else if (strcmp(KEY_MODEL_FILE, key) == 0) {
            model_file = val;
        } else if (strcmp(KEY_TRAINED_FILE, key) == 0) {
            weights_file = val;
        }
    }

    if (verbose) cout << "gpuid: " << gpuid << ", class_num: " << class_num << endl;
    if (verbose) cout << "max_size: " << max_size << ", scale_size: " << scale_size << endl;
    if (verbose) cout << "conf_thresh: " << conf_thresh << ", nms_thresh: " << nms_thresh << endl;
    if (verbose) cout << "model_file: " << model_file << endl;
    if (verbose) cout << "weights_file: " << weights_file << endl;

    assert(class_num > 1);
    assert(conf_thresh > 0 && conf_thresh < 1);
    assert(nms_thresh > 0 && nms_thresh < 1);
    assert(max_size > 0);
    assert(scale_size > 0 && scale_size < max_size);

    if (!checkFileExist(model_file.c_str())) {
        cout << "Error: model_file does not exist. Please set correct file path" << endl;
        return -1;
    }
    if (!checkFileExist(weights_file.c_str())) {
        cout << "Error: weights_file does not exist. Please set correct file path" << endl;
        return -1;
    }
	if (!checkDirExist(imgdir.c_str())) {
        cout << "Error: imgdir does not exist. Please set correct path." << endl;
        return -1;
    }
    tryMakedir(outdir.c_str());
	
    char imgDir[256] = {0};
	strcpy(imgDir, imgdir.c_str());
    char outDir[256] = {0};
    strcpy(outDir, outdir.c_str());

    tryAddSlash(imgDir);
    tryAddSlash(outDir);

    Caffe::SetDevice(gpuid);
    Caffe::set_mode(Caffe::GPU);// just test in GPU mode

    Detector det = Detector(model_file, weights_file,
        class_num, max_size, scale_size, conf_thresh, nms_thresh);

    struct dirent **imagenames;
    int n = scandir(imgdir.c_str(), &imagenames, filterImg, alphasort);
    if (n <= 0) {
        cout << "Error: no image files in " << imgdir << endl;
        return -1;
    }

    for (int i = 0; i < n; i++) {
        char *imgname = imagenames[i]->d_name;
        char fullname[512] = {0};
        sprintf(fullname, "%s%s", imgDir, imgname);
        cout << "------------------------------" << endl;
        cout << "Detect image " << fullname << endl;
        cv::Mat im = cv::imread(fullname);

        vector<vector<float> > ans;
        ans = det.Detect(im);
        for(int i = 0; i < ans.size(); i++) {
            if (verbose) {
                for(int j = 0; j < ans[i].size(); j++){
                    cout << ans[i][j] << " ";
                }
                cout << endl;
            }
            if (verbose) cout << "class name: " << class_names[ans[i][4]] << endl;
            if (verbose) cout << "score: " << ans[i][5] << endl;
            //ans[i][0] is xmin, ans[i][1] is ymin, ans[i][2] is width, ans[i][3] is height
            //ans[i][4] is class type, ans[i][5] is score
            drawRectOnImage(im, ans[i]);

            if (showlabel) {
                int offset = drawTextOnImage(im, ans[i], 0, class_names[ans[i][4]]);
                int margin = 2;
                drawTextOnImage(im, ans[i], offset + margin, to_string(ans[i][5]));
            }
        }
        char outname[512] = {0};
        sprintf(outname, "%slabled_%s", outDir, imgname);
        cout << "Save image with label: " << outname << endl;
        imwrite(outname, im);
        free(imagenames[i]);
    }
    free(imagenames);
    return 0;
}
