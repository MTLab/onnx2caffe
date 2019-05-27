# Convert pytorch to Caffe by ONNX
This tool converts [pytorch](https://github.com/pytorch/pytorch) model to [Caffe](https://github.com/BVLC/caffe) model by [ONNX](https://github.com/onnx/onnx)  
only use for inference

### Dependencies
* caffe (with python support)
* pytorch 0.4 (optional if you only want to convert onnx)
* onnx  

we recomand using protobuf 2.6.1 and install onnx from source  
```
git clone --recursive https://github.com/onnx/onnx.git
cd onnx 
python setup.py install
```

### How to use
run test.py to make sure it has been installed correctly  
To convert onnx model to caffe:
```
python convertCaffe.py ./model/MobileNetV2.onnx ./model/MobileNetV2.prototxt ./model/MobileNetV2.caffemodel
```
### Current support operation
* Conv
* ConvTranspose
* BatchNormalization
* MaxPool
* AveragePool
* Relu
* Sigmoid
* Dropout
* Gemm (InnerProduct only)
* Add
* Mul
* Reshape
* Upsample
* Concat
* Flatten
* Slice

### TODO List
 - [ ] support all onnx operations (which is impossible)
 - [ ] merge batchnormization to convolution
 - [ ] merge scale to convolution
