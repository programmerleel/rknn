import argparse
import os
import time
import sys
import numpy as np
import cv2
from rknn.api import RKNN



CLASSES = ("person", "bicycle", "car", "motorbike ", "aeroplane ", "bus ", "train", "truck ", "boat", "traffic light",
           "fire hydrant", "stop sign ", "parking meter", "bench", "bird", "cat", "dog ", "horse ", "sheep", "cow", "elephant",
           "bear", "zebra ", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
           "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife ",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza ", "donut", "cake", "chair", "sofa",
           "pottedplant", "bed", "diningtable", "toilet ", "tvmonitor", "laptop	", "mouse	", "remote ", "keyboard ", "cell phone", "microwave ",
           "oven ", "toaster", "sink", "refrigerator ", "book", "clock", "vase", "scissors ", "teddy bear ", "hair drier", "toothbrush ")

# 转换模型
def onnx2rknn(onnx_path,out_dir,out_name):
    rknn = RKNN(verbose=False)
    # load config
    print('loading config')
    ret = rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])
    if ret != 0:
        print("load config failed!")
        exit(ret)
    print("load config done!")
    # load model
    print('loading model')
    ret = rknn.load_onnx(onnx_path)
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

    rknn.accuracy_analysis(inputs=["/home/ubuntu/repositories/rknn/yolov8/python/images/bus.jpg"],output_dir="/home/ubuntu/repositories/rknn/yolov8/python/results")

    # export model
    print("exporting model")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    ret = rknn.export_rknn(os.path.join(out_dir,out_name))
    if ret != 0:
        print('export model failed!')
        exit(ret)
    print('export model done!')
    return rknn


# 缩放输入图片
def letterbox(image,image_size):
    height,width,_ = image.shape
    len = max(height,width)
    ratio = len/image_size
    h,w = int(height/ratio),int(width/ratio)
    top = (image_size-h)//2
    bottom = image_size-h-top
    left = (image_size-w)//2
    right = image_size-w-left
    image_resize = cv2.resize(image,(w,h))
    image_letter = cv2.copyMakeBorder(image_resize, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0,0,0))
    return image_letter


# 缩放图片转换为输入格式 HWC-->NCHW
def convert_input(image):
    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_chw = np.transpose(image_rgb,(2,0,1))
    # image_divide = image_chw/255.0
    # image_input = np.expand_dims(image_divide,axis=0).astype(np.float32)
    # return image_input
    return image_chw

# 转换推理结果格式
def convert_output(output):
    output_squeeze = np.squeeze(output,axis=0)
    output_trans = output_squeeze.transpose(1,0)
    return output_trans

def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

# 后处理
# 先判断置信度 减轻nms压力
def sift_conf(output,conf_thresh):
    conf = np.where(np.max(output[...,4:],axis=1,keepdims=True)>conf_thresh,True,False)
    conf_repeat = np.repeat(conf,output.shape[1],1)
    output_conf = output[conf_repeat].reshape(-1,84)
    return output_conf

# nms 处理框重叠问题 
def nms(output,nms_thresh):
    cls = list(set(np.argmax(output[...,4:],axis=1).tolist()))
    scores = np.max(output[...,4:],axis=1)
    boxes = xywh2xyxy(output[...,:4])
    print(boxes.shape)
    classes = np.argmax(output[...,4:],axis=1)
    # 分类处理
    nboxes, nclasses, nscores = [], [], []
    for c in cls:
        inds = np.where(classes == c,True,False)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]    
        x = b[:, 0]
        y = b[:, 1]
        w = b[:, 2] - b[:, 0]
        h = b[:, 3] - b[:, 1]
        area = w*h
        order = s.argsort()
        keep = []
        while order.size>0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1)
            h1 = np.maximum(0.0, yy2 - yy1)
            inter = w1 * h1
            ovr = inter / (area[i] + area[order[1:]] - inter)
            inds = np.where(ovr <= nms_thresh)[0]
            order = order[inds + 1]
        keep = np.array(keep)
        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])
    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)
    return boxes, classes, scores

def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = box
        print('class: {}, score: {}'.format(CLASSES[cl], score))
        print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))
        top = int(top)
        left = int(left)
        right = int(right)
        bottom = int(bottom)

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


def main(args):
    onnx_path = args.onnx_path
    out_dir = args.out_dir
    out_name = args.out_name
    image_path = args.image_path
    image_size = args.image_size
    conf_thresh = args.conf_thresh
    nms_thresh = args.nms_thresh
    rknn = onnx2rknn(onnx_path,out_dir,out_name)
    ret = rknn.init_runtime()
    print("===init environment===")
    if ret != 0:
        print("init environment failed!")
        exit(ret)
    print("init environment done!")
    print("===pre process==")
    t = time.time()
    image = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),cv2.IMREAD_COLOR)
    # image_letter = letterbox(image,image_size)[0]
    # image_input = convert_input(image_letter)
    image_resize = cv2.resize(image,(image_size,image_size))
    image_input = convert_input(image_resize)
    pre_t = time.time()-t
    print("pre process done!")
    print("===infer===")
    t = time.time()
    output = rknn.inference([image_input],data_format="nchw")[0]
    infer_t = time.time()-t
    print("infer done!")
    print("===post process===")
    t = time.time()
    output_convert = convert_output(output)
    output_conf = sift_conf(output_convert,conf_thresh)
    boxes,classes,scores = nms(output_conf,nms_thresh)
    draw(image,boxes,scores,classes)
    post_t = time.time()-t
    print("pre process time:{} infer time:{} post process time:{} total time:{}".format(pre_t,infer_t,post_t,pre_t+infer_t+post_t))
    cv2.imshow("post process result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rknn.release()
    

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path',type=str,default="/home/ubuntu/repositories/rknn/yolov8/python/models/yolov8n.onnx")
    parser.add_argument('--out_dir',type=str,default="/home/ubuntu/repositories/rknn/yolov8/python/models")
    parser.add_argument('--out_name',type=str,default="yolov8n.rknn")
    parser.add_argument("--image_path",type=str,help="",default="/home/ubuntu/repositories/rknn/yolov8/python/images/bus.jpg")
    parser.add_argument("--image_size",type=float,help="",default=640)
    parser.add_argument("--conf_thresh",type=float,help="",default=0.5) 
    parser.add_argument("--nms_thresh",type=float,help="",default=0.6)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
    