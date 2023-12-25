import argparse
import os
from rknn.api import RKNN
import sys


def onnx2rknn(onnx_path,out_dir,out_name):
    rknn = RKNN()
    # load config
    print('loading config')
    ret = rknn.config(target_platform='rk3588')
    if ret != 0:
        print("load config failed!")
        exit(ret)
    print("load config done!")
    # load model
    print('loading model')
    ret = rknn.load_onnx(onnx_path,inputs=["images"],input_size_list=[[640,640]],outputs=['/model.22/Mul_2_output_0', '/model.22/Split_output_1'])
    if ret != 0:
        print("load model failed!")
        exit(ret)
    print("load model done!")    
    # build model
    print('building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print("build model failed!")
        exit(ret)
    print("build model done!")   
    # export model
    print("exporting model")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    ret = rknn.export_rknn(os.path.join(out_dir,out_name))
    if ret != 0:
        print('export model failed!')
        exit(ret)
    print('export model done!')


def main(args):
    onnx_path = args.onnx_path
    out_dir = args.out_dir
    out_name = args.out_name
    onnx2rknn(onnx_path,out_dir,out_name)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path',type=str,default="/home/ubuntu/repositories/rknn/yolov8/python/models/yolov8n.onnx")
    parser.add_argument('--out_dir',type=str,default="/home/ubuntu/repositories/rknn/yolov8/python/models")
    parser.add_argument('--out_name',type=str,default="yolov8n.rknn")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))