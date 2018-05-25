from __future__ import print_function
import onnx
import numpy as np
import caffe
caffe.set_mode_cpu()
import importlib
from convertCaffe import convertToCaffe, getGraph
import os

def getPytorchModel(name):
    py_model_path = 'model'
    module = importlib.import_module("model_generator."+name)
    var, model = module.get_model_and_input(py_model_path)
    return var, model

module_name_list = [
                    "broadcast_mul",
                    "broadcast_add",
                    "googlenet",
                    "resnet",
                    "MobileNetV2",
                    ]

model_save_dir = 'model'
if not os.path.isdir(model_save_dir):
    os.makedirs(model_save_dir)

for module_name in module_name_list:
    print("export {} onnx model ...".format(module_name))
    module = importlib.import_module("model_generator"+"."+module_name)
    module.export(model_save_dir)

    var, pt_model = getPytorchModel(module_name)
    var_numpy = var.data.numpy()
    pt_model.eval()
    pt_out = pt_model(var)
    pt_out = pt_out.data.numpy()
    onnx_path = os.path.join("model", module_name+'.onnx')
    prototxt_path = os.path.join("model", module_name+'.prototxt')
    caffemodel_path = os.path.join("model", module_name+'.caffemodel')

    graph = getGraph(onnx_path)
    print("converting {} to caffe ...".format(module_name))
    caffe_model = convertToCaffe(graph, prototxt_path, caffemodel_path)

    input_name = str(graph.inputs[0][0])
    output_name = str(graph.outputs[0][0])

    caffe_model.blobs[input_name].data[...] = var_numpy
    net_output = caffe_model.forward()
    caffe_out = net_output[output_name]

    minus_result = caffe_out-pt_out
    mse = np.sum(minus_result*minus_result)

    print("{} mse between caffe and pytorch model output: {}".format(module_name,mse))



    

    

