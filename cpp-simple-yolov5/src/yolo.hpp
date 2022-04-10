


#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#	include <dirent.h>
#	include <sys/types.h>
#	include <sys/stat.h>
#	include <unistd.h>
#   include <stdarg.h>

#include "simple_yolo.hpp"
using namespace std;

bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}
class yolo
{
public:
    shared_ptr<SimpleYolo::Infer> engine;
    vector<shared_future<SimpleYolo::BoxArray>> boxes_array;
    bool init();
    vector<shared_future<SimpleYolo::BoxArray>> pross_image(cv::Mat image);
};

bool yolo::init()
{
    int deviceid = 0;
    SimpleYolo::set_device(deviceid);
    cout<<"ks"<<endl;
    
    string engine_file="/home/long/home/project1/cpp-simple-yolov5/workspace/yolov5s_dynamic.FP32.trtmodel";
    string onnx_file="/home/long/home/project1/cpp-simple-yolov5/workspace/yolov5s_dynamic.onnx";

    int test_batch_size = 1;
    
    if(!exists(engine_file)){
        SimpleYolo::compile(
            SimpleYolo::Mode::FP32, SimpleYolo::Type::V5,                 // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            engine_file,                 // save to
            1 << 30,
            "inference"
        );
    }
    
    engine= SimpleYolo::create_infer(engine_file, SimpleYolo::Type::V5, deviceid, 0.4f, 0.5f);
   
    if(engine == nullptr){
        printf("Engine is nullptr\n");
        return false;
    }
    
    return true;
}

vector<shared_future<SimpleYolo::BoxArray>> yolo::pross_image(cv::Mat image)
{   
    vector<cv::Mat> images;
    images.emplace_back(image);
    boxes_array=engine->commits(images);
    boxes_array.back().get();
    return boxes_array;

}
