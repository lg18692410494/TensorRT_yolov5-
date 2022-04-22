#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


#include <iostream>
#include <algorithm>
#include <fstream>
#include <memory>

#include <future>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <queue>
#include <string>
#include <functional>
#include <sstream>
#if defined(_WIN32)
#	include <Windows.h>
#   include <wingdi.h>
#	include <Shlwapi.h>
#	pragma comment(lib, "shlwapi.lib")
#	undef min
#	undef max
#else
#	include <dirent.h>
#	include <sys/types.h>
#	include <sys/stat.h>
#	include <unistd.h>
#   include <stdarg.h>
#endif
using namespace std;

static double timestamp_now_float() {
    return chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
}
static bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}
//仿射变换矩阵
struct AffineMatrix{
    float yuan[6];       // image to dst(network), 2x3 matrix 原矩阵
    float ni[6];       // dst to image, 2x3 matrix      逆矩阵

    void compute(cv::Size& origin,cv::Size& dst)
    {
        float scale_x = dst.width / (float)origin.width;
        float scale_y = dst.height / (float)origin.height;
        float scale=std::min(scale_x,scale_y);

        yuan[0]=scale; yuan[1]=0; yuan[2]= -scale*origin.width*0.5+dst.width*0.5;
        yuan[3]=0;yuan[4]=scale;yuan[5]=-scale*origin.height*0.5+dst.height*0.5;

        cv::Mat m2x3_yuan(2, 3, CV_32F, yuan);
        cv::Mat m2x3_ni(2, 3, CV_32F, ni);
        cv::invertAffineTransform(m2x3_yuan, m2x3_ni);//求其逆矩阵
    }
    };
    //框
struct Box{
    float left, top, right, bottom, confidence;
    int label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label):
    left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
};

bool AFF_resize(cv::Mat& src,float* dst_dev,int size,AffineMatrix& MM,cudaStream_t stream);

void decode_kernel_invoker(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream);
inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kINFO){
         
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} ;
static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};
class yolo
{


    public:
    TRTLogger gLogger;
    nvinfer1::ICudaEngine* engine=nullptr;
    nvinfer1::IExecutionContext* execution_context=nullptr;
    cudaStream_t stream = nullptr;
    
    void *buffers[2];
    std::vector<int64_t> bufferSize;
    float conf_threshold=0.45;
    float nms_thres=0.45;
    int BATCH_SIZE=1;
    int d_size=640;
    int nelem;
    int nc=80;
    int ncols = 5+nc;
    

    
    

bool build_model(string source_onnx,string save_engine,string mode);
   
bool load_engine(string engine_file);

vector<Box> infer(cv::Mat image);

void init()
{
    string onnx_w="";
    string engine_w="";
    if(!exists(engine_w))
        build_model(onnx_w,engine_w,"FP16");
    load_engine(engine_w);

}

yolo()
{
    cout<<"create succese"<<endl;


}
~yolo()
{

    (cudaStreamDestroy(stream));
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    execution_context->destroy();
    engine->destroy();
    cout<<"end!!!"<<endl;

}

void destroy()
{

(cudaStreamDestroy(stream));
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    execution_context->destroy();
    engine->destroy();
    cout<<"end!!!"<<endl;

}



};



 