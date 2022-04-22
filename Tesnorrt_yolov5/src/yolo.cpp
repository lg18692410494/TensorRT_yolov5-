#include"yolo.h"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include<string>
using namespace nvinfer1;
using namespace nvonnxparser;
using namespace std;



template<typename _T>
static void destroy_nvidia_pointer(_T* ptr) {
        if (ptr) ptr->destroy();
    }
static bool save_file(const string& file, const void* data, size_t length){

        FILE* f = fopen(file.c_str(), "wb");
        if (!f) return false;

        if (data && length > 0){
            if (fwrite(data, 1, length, f) != length){
                fclose(f);
                return false;
            }
        }
        fclose(f);
        return true;
    }
vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}

vector<Box> gpu_decode(float* predict_device, int rows, int cols,float* invert_affine_matrix ,cudaStream_t stream,float confidence_threshold = 0.25f, float nms_threshold = 0.45f){
    
    vector<Box> box_result;
    float *warp_affine_matrix_2_3=NULL;

    float* output_device = NULL;
    float* output_host = NULL;

    int max_objects = 1000;
    int NUM_BOX_ELEMENT = 7;  // left, top, right, bottom, confidence, class, keepflag

    (cudaMalloc((void**)&warp_affine_matrix_2_3,sizeof(float)*6));
    (cudaMalloc(&output_device, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    
    cudaMemset(output_device,0,sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float));

    (cudaMallocHost(&output_host, sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float)));
    //cudaMemset(output_host,0,sizeof(float) + max_objects * NUM_BOX_ELEMENT * sizeof(float));

    (cudaMemcpy(warp_affine_matrix_2_3, invert_affine_matrix, sizeof(float)*6 ,cudaMemcpyHostToDevice));

   // (cudaMemcpyAsync(predict_device, predict, rows * cols * sizeof(float), cudaMemcpyHostToDevice, stream));
    decode_kernel_invoker(
        predict_device, rows, cols - 5, confidence_threshold, 
        nms_threshold, warp_affine_matrix_2_3, output_device, max_objects, NUM_BOX_ELEMENT, stream
    );
    (cudaMemcpyAsync(output_host, output_device, 
        sizeof(int) + max_objects * NUM_BOX_ELEMENT * sizeof(float), 
        cudaMemcpyDeviceToHost, stream
    ));
    (cudaStreamSynchronize(stream));

    int num_boxes = min((int)output_host[0], max_objects);
   
    for(int i = 0; i < num_boxes; ++i){
        float* ptr = output_host + 1 + NUM_BOX_ELEMENT * i;
        int keep_flag = ptr[6];
        if(keep_flag){
            box_result.emplace_back(
                ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], (int)ptr[5]
            );
        }
    }
    (cudaFree(output_device));
    (cudaFree(warp_affine_matrix_2_3));
    (cudaFreeHost(output_host));
    return box_result;
}
   
bool yolo::build_model( string source_onnx,
                     string save_engine,string mode)
{
     int max_workspace_size=1;
    int max_batch_size=1;
     shared_ptr<IBuilder> builder1(createInferBuilder(gLogger), destroy_nvidia_pointer<IBuilder>);
        if (builder1 == nullptr) {
            printf("Can not create builder.\n");
            return false;
        }
   
    shared_ptr<IBuilderConfig> config(builder1->createBuilderConfig(), destroy_nvidia_pointer<IBuilderConfig>);
    if (mode=="FP16") {//设置FP16
        if (!builder1->platformHasFastFp16()) {
            printf("Platform not have fast fp16 support\n");
        }
        config->setFlag(BuilderFlag::kFP16);
    }
   
    shared_ptr<INetworkDefinition> network;
    shared_ptr<nvonnxparser::IParser> onnxParser;
    const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    
    network = shared_ptr<INetworkDefinition>(builder1->createNetworkV2(explicitBatch), destroy_nvidia_pointer<INetworkDefinition>);
  
  
    //from onnx is not markOutput
    onnxParser.reset(nvonnxparser::createParser(*network, gLogger), destroy_nvidia_pointer<nvonnxparser::IParser>);
    //auto onnxParser = nvonnxparser::createParser(*network,gLogger);
    if (onnxParser == nullptr) {
        printf("Can not create parser.\n");
        return false;
    }
    if (!onnxParser->parseFromFile(source_onnx.c_str(), 1)) {
        printf("Can not parse OnnX file: %s\n", source_onnx.c_str());
        return false;
    }
    auto inputTensor = network->getInput(0);
    auto inputDims = inputTensor->getDimensions();
    int net_num_input = network->getNbInputs();
    printf("Network has %d inputs:\n", net_num_input);
    vector<string> input_names(net_num_input);
    for(int i = 0; i < net_num_input; ++i){
        auto tensor = network->getInput(i);
        auto dims = tensor->getDimensions();
        for(int i=0;i<4;i++)
            printf("%d ",dims.d[i]);
        printf("\n");
        input_names[i] = tensor->getName();
    }
    int net_num_output = network->getNbOutputs();
    printf("Network has %d outputs:\n", net_num_output);
    for(int i = 0; i < net_num_output; ++i){
        auto tensor = network->getOutput(i);
        auto dims = tensor->getDimensions();
        for(int i=0;i<3;i++)
            printf("%d ",dims.d[i]);   
        printf("\n");
    }
    int net_num_layers = network->getNbLayers();
    printf("Network has %d layers\n", net_num_layers);		
    builder1->setMaxBatchSize(max_batch_size);
    config->setMaxWorkspaceSize(max_workspace_size);
    auto profile = builder1->createOptimizationProfile();
    for(int i = 0; i < net_num_input; ++i){
        auto input = network->getInput(i);
        auto input_dims = input->getDimensions();
        input_dims.d[0] = 1;
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
        input_dims.d[0] = max_batch_size;
        profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    }
    config->addOptimizationProfile(profile);
    printf("Building engine...\n");
  
    shared_ptr<ICudaEngine> engine(builder1->buildEngineWithConfig(*network, *config), destroy_nvidia_pointer<ICudaEngine>);
    if (engine == nullptr) {
        printf("engine is nullptr\n");
        return false;
    }
    shared_ptr<IHostMemory> seridata(engine->serialize(), destroy_nvidia_pointer<IHostMemory>);

    save_file(save_engine, seridata->data(), seridata->size()); 
   
    return true;

}

   
bool yolo::load_engine(string engine_file)
{
    cudaStreamCreate(&stream);
    auto engine_data = load_file(engine_file);
    nvinfer1::IRuntime* runtime   = nvinfer1::createInferRuntime(gLogger);
    engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());
    if(engine == nullptr){
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return false; 
    }
    execution_context = engine->createExecutionContext();
        int nbBindings = engine->getNbBindings();
    bufferSize.resize(nbBindings);
    for (int i = 0; i < nbBindings; ++i) {
        // 获取输入或输出的维度信息
        nvinfer1::Dims dims = engine->getBindingDimensions(i);
        // 获取输入或输出的数据类型信息
        nvinfer1::DataType dtype = engine->getBindingDataType(i);
        int64_t totalSize=1;
        for(int k=0;k<dims.nbDims;k++)
            totalSize*=dims.d[k];
        totalSize*=sizeof(dtype);
        //int64_t totalSize = volume(dims) * 1 * getElementSize(dtype);
        bufferSize[i] = totalSize;
        std::cout << "binding" << i << ": " << totalSize << std::endl;
        // &buffers是双重指针 相当于改变指针本身，这里就是把输入或输出进行向量化操作
        cudaMalloc(&buffers[i], totalSize);
    }
    nelem = bufferSize[1] / sizeof(float);
    return true;

}

vector<Box> yolo::infer(cv::Mat image)
{
    AffineMatrix MM;//仿射变换矩阵
   
    if(!AFF_resize(image,(float*)buffers[0],d_size,MM,stream))
       printf("error");
   // printf("MM:yuan:\n%f %f %f \n %f %f %f\n",MM.yuan[0],MM.yuan[1],MM.yuan[2],MM.yuan[3],MM.yuan[4],MM.yuan[5]);
   // printf("MM:ni:\n%f %f %f \n %f %f %f\n",MM.ni[0],MM.ni[1],MM.ni[2],MM.ni[3],MM.ni[4],MM.ni[5]);
   
    execution_context->execute(BATCH_SIZE, buffers);
    cudaStreamSynchronize(stream);
    int nrows = nelem / ncols;
    auto boxes = gpu_decode((float*)buffers[1], nrows, ncols,MM.ni,stream,conf_threshold,nms_thres);
    return boxes;



}