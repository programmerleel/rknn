#include <iostream>
#include <stdio.h>
#include <include/rknn_api.h>
#include <memory.h>

void *model_path;

int main(){
    int ret = 0;
    int channel = 0;
    int height = 0;
    int width = 0;
    // 加载模型 模型的二进制数据或者模型的路径
    rknn_context ctx;
    ret = rknn_init(&ctx,model_path,0,0,NULL);
    // 获取模型相关信息
    // 查询sdk版本
    rknn_sdk_version sdk_version;
    ret = rknn_query(ctx,RKNN_QUERY_SDK_VERSION,&sdk_version,sizeof(sdk_version));
    if(ret!=0){
      printf("get sdk version failed!\n");
      return ret;
    }
    // 使用cout会出现运算符不匹配的问题
    printf("sdk api version: %s\n", sdk_version.api_version);
    // 查询输入输出的tensor个数
    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if(ret!=0){
      printf("get num of input and output failed!");
      return ret;
    } 
    printf("input tensor num:%d output tensor num:%d\n",io_num.n_input,io_num.n_output);
    // 查询输入tensor的属性
    rknn_tensor_attr input_attrs[io_num.n_input];
    // 将某一块内存全部初始化为制定值
    memset(input_attrs, 0, io_num.n_input*sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_input; i++) {
    input_attrs[i].index = i;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]),sizeof(rknn_tensor_attr));
      if(ret!=0){
        printf("get input tensor info failed!");
        return ret;
      } 
      if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
      channel = input_attrs[0].dims[1];
      height  = input_attrs[0].dims[2];
      width   = input_attrs[0].dims[3];
      printf("input tensor channel:%d input tensor height:%d input tensor width:%d\n",channel,height,width);
    }
    }
    // 查询输出tensor的属性
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, io_num.n_output * sizeof(rknn_tensor_attr));
    for (int i = 0; i < io_num.n_output; i++) {
      output_attrs[i].index = i;
      ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
      if (ret != 0) {
        printf("get output tensor info failed!\n");
        return ret;
      }
      // printf("output tensor dim:%n output tensor size:%n\n",output_attrs[i].dims,output_attrs[i].size);
    }
    // 设置输入 这个api在rv1103/rv1106上不支持
    
    // rknn_input inputs[io_num.n_input];
    // memset(inputs, 0, sizeof(inputs));
    // inputs[0].index = 0;
    // inputs[0].buf = input_data;
    // inputs[0].size = state_dim * sizeof(float);
    // inputs[0].pass_through = 0;
    // inputs[0].type = RKNN_TENSOR_FLOAT16;
    // inputs[0].fmt = RKNN_TENSOR_NHWC;
    // ret = rknn_inputs_set(ctx, io_num.n_input, inputs);


    // 查询模型的推理时间 不包含设置输入输出的时间
    rknn_perf_run infer_time;
    ret = rknn_query(ctx,RKNN_QUERY_PERF_RUN,&infer_time,sizeof(infer_time));

}