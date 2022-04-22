
#include"yolo.h"

cv::Mat image_test(cv::Mat &image,float new_wh=640.0)
{

    //cv::Mat image=cv::imread("/home/long/home/project1/QDJYY_transit_assembly_train_20220316_4.jpg");
    int h=image.rows;
    int w=image.cols;
    
    float scale=min(new_wh/h,new_wh/w);
    int nh=h*scale;
    int nw=w*scale;
   
    cv::Mat new_image;
    // auto time1 = timestamp_now_float();
    cv::resize(image,new_image,cv::Size(nw,nh),0,0,cv::INTER_LINEAR);
    // auto time2 = timestamp_now_float()-time1;
    // cout<<"using_time: "<<time2<<endl;
    //cv::imshow("1",new_image);
    //cv::imshow("old",image);
    //cv::waitKey(0);
    return new_image;

}
void viode_test(string video_path,yolo& yolo)
{
  
    cv::VideoCapture capture;
    capture.open(video_path);
    if(!capture.isOpened())
    {
        std::cout << "打开视频失败" << std::endl;
        return ;
    }
   
    int fps = capture.get(cv::CAP_PROP_FPS);
    cout<<"video fps:"<<fps<<endl;
    cv::Mat image;
    auto time1 = timestamp_now_float();
    while(capture.read(image))
    {   auto begin_timer = timestamp_now_float();
        cv::Mat frame=image_test(image);
        auto boxes=yolo.infer(frame); 
        for(auto& obj : boxes){
            //printf("%f %f %f %f\n",obj.left,obj.right,obj.top,obj.bottom);
            uint8_t b, g, r;
            cv::rectangle(frame, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 2);
            auto name    = cocolabels[obj.label];
            auto caption = cv::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(frame, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(frame, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
            
        }
        float inference_time = (timestamp_now_float() - begin_timer);
        int FPS=1000/inference_time;
        auto cap_fps = cv::format("FPS: %d",FPS);
        cv::putText(frame, cap_fps, cv::Point(30,30), 0, 1, cv::Scalar(0,0,255), 2, 16);
         auto box_size = cv::format("nums: %d",boxes.size());
        cv::putText(frame, box_size, cv::Point(170,30), 0, 1, cv::Scalar(0,0,255), 2, 16);
        cv::imshow("mp4",frame);
        char c=cv::waitKey(1);
        if(c==27)
            break;
    }

    capture.release();
    auto time2 = timestamp_now_float()-time1;
    cout<<"all_using_time: "<<time2<<endl;
}



int main() {
    //image_test();
    yolo yolos;
    //yolo.build_model("/home/long/home/project1/Tensorrt_test/workspace/yolov5n.onnx","yolov5n_fp16.engine","FP16");
    yolos.load_engine("/home/long/home/project1/Tensorrt_test/workspace/yolov5n.engin");

    viode_test("/home/long/home/project1/xr.mp4",yolos);

  /*   cv::Mat image=cv::imread("/home/long/home/project1/Tensorrt_test/workspace/car.jpg");
    
    for(int i=0;i<50;i++)
    {   auto t1=timestamp_now_float();
        auto boxes=yolo.infer(image);
        auto T=timestamp_now_float()-t1;
        cout<<"using_time: "<<T<<endl;
        for(auto box:boxes)
        {
            printf("%f %f %f %f\n",box.left,box.right,box.top,box.bottom);
            //cv::rectangle(image, cv::Point(int(box.left),int(box.top)), cv::Point(int(box.right), int(box.bottom)), cv::Scalar(0, 0,255), 2);
        }
    }
    cv::imshow("1",image);
    cv::waitKey(0); */
    return 0;
}


