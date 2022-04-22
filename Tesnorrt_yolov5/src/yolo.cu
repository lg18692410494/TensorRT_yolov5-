
#include"yolo.h"
using namespace std;
using namespace cv;
 
#define checkCudaKernel(...)                                                                         \
    __VA_ARGS__;                                                                                     \
    do{cudaError_t cudaStatus = cudaPeekAtLastError();                                               \
    if (cudaStatus != cudaSuccess){                                                                  \
        printf("launch failed: %s\n", cudaGetErrorString(cudaStatus));                                  \
    }} while(0); 




__global__ void warp_affine_resize(uint8_t* origin,float* dst,float* warp_affine_matrix_2_3,int dst_width, int dst_height,int orgin_width, int orgin_height)
{   uint8_t const_value_st=0;
    
    int dst_x = blockDim.x * blockIdx.x + threadIdx.x; 
    int dst_y = blockDim.y * blockIdx.y + threadIdx.y;
    if (dst_x >= dst_width || dst_y >= dst_height)  return;
  
    
    float ox=dst_x*warp_affine_matrix_2_3[0]+dst_y*warp_affine_matrix_2_3[1]+warp_affine_matrix_2_3[2];
    float oy=dst_x*warp_affine_matrix_2_3[3]+dst_y*warp_affine_matrix_2_3[4]+warp_affine_matrix_2_3[5];

    float c1=const_value_st;
    float c2=const_value_st;
    float c3=const_value_st;
    if(ox>=0 && ox<orgin_width && oy>=0 && oy<orgin_height)
    {   
        //没有越界
        int low_x=floorf(ox);
        int low_y=floorf(oy);
        int high_x=low_x+1;
        int high_y=low_y+1;
        //printf("%d %d\n",low_x,low_y);
        float pos_x=ox-low_x;
        float pos_y=oy-low_y;

        float p0_area=(1-pos_x)*(1-pos_y);
        float p1_area=pos_x*(1-pos_y);
        float p2_area=(1-pos_x)*pos_y;
        float p3_area=pos_x*pos_y;

        uint8_t const_value[] = {const_value_st, const_value_st, const_value_st};
        uint8_t* v1 = const_value;
        uint8_t* v2 = const_value;
        uint8_t* v3 = const_value;
        uint8_t* v4 = const_value;

        if(low_y>=0)
        {
            if(low_x>=0)
                {   v1=origin+(orgin_width*low_y+low_x)*3;
                    //int p=(orgin_width*low_y+low_x)*3;
                    //v1[0]=origin[p];
                    //v1[1]=origin[p+1];
                    //v1[2]=origin[p+2];

                }
            if(high_x<orgin_width)
                {
                    v2=origin+(orgin_width*low_y+high_x)*3;
                    //int p=(orgin_width*low_y+high_x)*3;
                    //v2[0]=origin[p];
                    //v2[1]=origin[p+1];
                    //v2[2]=origin[p+2];
                }
        }
        if(high_y<orgin_height){

            if(low_x>0)
            {
                v3=origin+(orgin_width*high_y+low_x)*3;
                /* int p=(orgin_width*high_y+low_x)*3;
                v3[0]=origin[p];
                v3[1]=origin[p+1];
                v3[2]=origin[p+2]; */
            }

            if(high_x<orgin_width)
            {
                v4=origin+(orgin_width*high_y+high_x)*3;
               /*  int p=(orgin_width*high_y+high_x)*3;
                    v4[0]=origin[p];
                    v4[1]=origin[p+1];
                    v4[2]=origin[p+2]; */
            }

        }

        c1=floorf(p0_area*v1[0]+p1_area*v2[0]+p2_area*v3[0]+p3_area*v4[0]);
        c2=floorf(p0_area*v1[1]+p1_area*v2[1]+p2_area*v3[1]+p3_area*v4[1]);
        c3=floorf(p0_area*v1[2]+p1_area*v2[2]+p2_area*v3[2]+p3_area*v4[2]);
    }
    
   
    int area=dst_width*dst_height;
    int dst_idx=(dst_width*dst_y+dst_x);
    dst[dst_idx]=c3/255.;
    dst[dst_idx+area]=c2/255.;
    dst[dst_idx+2*area]=c1/255.;
    
}


bool AFF_resize(cv::Mat& src,float* dst_dev,int size,AffineMatrix& MM,cudaStream_t stream)
{
  
    cv::Size dst_size(size,size);
    cv::Size origin_size=src.size();
    MM.compute(origin_size,dst_size);
    //cv::Mat dst =cv::Mat::zeros(dst_size, CV_8UC3);//创建图像

    //printf("MM:yuan:\n%f %f %f \n %f %f %f\n",MM.yuan[0],MM.yuan[1],MM.yuan[2],MM.yuan[3],MM.yuan[4],MM.yuan[5]);
    //printf("MM:ni:\n%f %f %f \n %f %f %f\n",MM.ni[0],MM.ni[1],MM.ni[2],MM.ni[3],MM.ni[4],MM.ni[5]);

    /* uint8_t *src_data = src.data;          // src_data指针指向图像src的数据
	if (!src.isContinuous())  return false;   //判断图像数据字节是否填充 */
 
	uint8_t *src_dev;
    float *warp_affine_matrix_2_3=NULL;
	int src_length = src.rows * src.cols * src.channels();
    //int dst_length=dst_size.width*dst_size.height*src.channels();
   // printf("rows:%d clos:%d channels:%d \n",src.rows,src.cols,src.channels());
	//在gpu上分配内存空间
	(cudaMalloc(&src_dev, src_length*sizeof(uint8_t)));

	//(cudaMalloc((void**)&dst_dev, dst_length*sizeof(uint8_t)));

    (cudaMalloc((void**)&warp_affine_matrix_2_3,sizeof(float)*6));
    
    //从cup拷贝到gpu
	(cudaMemcpy(src_dev, src.data, src_length*sizeof(uint8_t), cudaMemcpyHostToDevice));
    (cudaMemcpy(warp_affine_matrix_2_3, MM.ni, sizeof(float)*6 ,cudaMemcpyHostToDevice));

    /* dim3 block_0(32,16);
    dim3 grid_0((size-1)/block_0.x+1,(size-1)/block_0.y+1); */
    dim3 block_size(32, 32); // blocksize最大就是1024，这里用2d来看更好理解
    dim3 grid_size((dst_size.width + 31) / 32, (dst_size.height + 31) / 32);
    checkCudaKernel(warp_affine_resize<<<grid_size,block_size,0,stream>>>(src_dev,dst_dev,warp_affine_matrix_2_3,size,size,src.cols,src.rows));

    // printf("GPU Execution configuration<<<(%d,%d),(%d,%d)>>>\n", grid_0.x,grid_0.y,block_0.x,block_0.y);
	//将数组dst_dev复制至cpu
	//(cudaMemcpy(dst.data, dst_dev, dst_length*sizeof(uint8_t), cudaMemcpyDeviceToHost));
 
	//cudaFree(src_dev);
    cudaFree(warp_affine_matrix_2_3);
	//cudaFree(dst_dev);
 
    return true;
}

static __device__ void affine_project(float* matrix, float x, float y, float* ox, float* oy){
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __global__ void decode_kernel(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT
){  
    //parray保存boxs
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes) return;

    float* pitem     = predict + (5 + num_classes) * position;
    float objectness = pitem[4];
    if(objectness < confidence_threshold)
        return;

    float* class_confidence = pitem + 5;
    float confidence        = *class_confidence++;
    int label               = 0;
    for(int i = 1; i < num_classes; ++i, ++class_confidence){
        if(*class_confidence > confidence){
            confidence = *class_confidence;
            label      = i;
        }
    }

    confidence *= objectness;
    if(confidence < confidence_threshold)
        return;

    int index = atomicAdd(parray, 1);//第一位为boxes总数，原子操作（不可打断）自动加一
    if(index >= max_objects)
        return;
    //printf("index:%d\n",index);
    float cx         = *pitem++;
    float cy         = *pitem++;
    float width      = *pitem++;
    float height     = *pitem++;
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left,  top,    &left,  &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    // left, top, right, bottom, confidence, class, keepflag
    float* pout_item = parray + 1 + index * NUM_BOX_ELEMENT;
    *pout_item++ = left;
    *pout_item++ = top;
    *pout_item++ = right;
    *pout_item++ = bottom;
    *pout_item++ = confidence;
    *pout_item++ = label;
    *pout_item++ = 1; // 1 = keep, 0 = ignore
}

static __device__ float box_iou(
    float aleft, float atop, float aright, float abottom, 
    float bleft, float btop, float bright, float bbottom
){

    float cleft 	= max(aleft, bleft);
    float ctop 		= max(atop, btop);
    float cright 	= min(aright, bright);
    float cbottom 	= min(abottom, bbottom);
    
    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if(c_area == 0.0f)
        return 0.0f;
    
    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

static __global__ void fast_nms_kernel(float* bboxes, int max_objects, float threshold, int NUM_BOX_ELEMENT){

    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count = min((int)*bboxes, max_objects);//box 个数
    
    if (position >= count) 
        return;
    //printf("(int)*bboxes:%d\n",(int)*bboxes);
    // left, top, right, bottom, confidence, class, keepflag
    //   0    1     2       3         4         5       6
    float* pcurrent = bboxes + 1 + position * NUM_BOX_ELEMENT;
    for(int i = 0; i < count; ++i){
        float* pitem = bboxes + 1 + i * NUM_BOX_ELEMENT;
        if(i == position || pcurrent[5] != pitem[5]||pitem[6]==0) continue;//同一个或者类别不同跳过

        if(pitem[4] >= pcurrent[4]){
            if(pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou = box_iou(
                pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
                pitem[0],    pitem[1],    pitem[2],    pitem[3]
            );

            if(iou > threshold){
                pcurrent[6] = 0;  // 1=keep, 0=ignore
                return;
            }
        }
    }
} 

void decode_kernel_invoker(
    float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
    float nms_threshold, float* invert_affine_matrix, float* parray, int max_objects, int NUM_BOX_ELEMENT, cudaStream_t stream){
    //NUM_BOX_ELEMENT=7
    //max_objects=1000
    auto block = num_bboxes > 512 ? 512 : num_bboxes;
    auto grid = (num_bboxes + block - 1) / block;

    
   // printf("block:%d grid:%d\n",block,grid);
    /* 如果核函数有波浪线，没关系，他是正常的，你只是看不顺眼罢了 */
    decode_kernel<<<grid, block, 0, stream>>>(
        predict, num_bboxes, num_classes, confidence_threshold, 
        invert_affine_matrix, parray, max_objects, NUM_BOX_ELEMENT
    );
    
    block = max_objects > 512 ? 512 : max_objects;
    grid = (max_objects + block - 1) / block;
    //printf("block:%d grid:%d\n",block,grid);
    fast_nms_kernel<<<grid, block, 0, stream>>>(parray, max_objects, nms_threshold, NUM_BOX_ELEMENT);
}