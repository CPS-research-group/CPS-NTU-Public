'''
Model Conversion for Optical Flow Models

'''
import importlib
import os
import sys
import tensorflow as tf
import numpy as np
from torch import nn
import torch
import torch.onnx
import cv2
import argparse

# Optical Flow Model for TPU
from safe_ml.training.of_optprior_network import Encoder

MODELOPTIMIZERPATH = "/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py"


class OFModel(nn.Module):
    def __init__(self, modelfile):
        super(OFModel, self).__init__()

        device = torch.device("cpu")
        self.config = {}
        for (key, value) in np.loadtxt(modelfile + "_config",
                                       delimiter=":",
                                       comments="#", dtype=str):
            self.config[key] = float(value)
        # Image Size (HxW) after model type e.g. bi3dof_120x160_113x152.
        self.image_size = [int(i) for i in modelfile.split(
            '.')[-2].split('_')[1].split('x')]
        print(f"Image Size: {self.image_size}..")
        # Cropped Image Size (HxW) After Image Size e.g. bi3dof_120x160_113x152.
        self.cropped_size = [int(i) for i in modelfile.split(
            '.')[-2].split('_')[2].split('x')]
        print(f"Cropped Image Size: {self.cropped_size}..")
        self.crop_width1 = int(
            (self.image_size[1] - self.cropped_size[1]) / 2)
        if ((self.image_size[1] - self.cropped_size[1])) % 2 != 0:
            print("Width difference is odd")
            self.crop_width2 = self.crop_width1 + 1
        else:
            self.crop_width2 = self.crop_width1

        self.crop_height1 = int(
            (self.image_size[0] - self.cropped_size[0]) / 2)
        if ((self.image_size[0] - self.cropped_size[0])) % 2 != 0:
            print("Height difference is odd")
            self.crop_height2 = self.crop_height1 + 1
        else:
            self.crop_height2 = self.crop_height1
        self.encoder = Encoder(device, self.config, self.cropped_size)
        self.load_state_dict(torch.load(modelfile, map_location=device))
        self.eval()


def rep_data_gen(imagefile="../typical_datasets/typicalset_duckietown_flows_train_id.npy"):
    # [Images, C, H, W]
    of_array = np.load(imagefile, allow_pickle=True)
    for frame in of_array:
        # print(frame.shape) # (2, 120, 160)
        frame = np.transpose(frame, (1, 2, 0))  # Convert to HWC
        frame = cv2.resize(frame, (model.image_size[1], model.image_size[0]))
        frame = np.transpose(frame, (2, 0, 1))  # Convert to CHW
        frame = frame[:, model.crop_height1:model.image_size[0]-model.crop_height2,
                      model.crop_width1:model.image_size[1]-model.crop_width2]  # Crop
        # x = np.zeros((2, 113, 152, 1))		# Depth 1 for OF Model
        # Depth 6 for OF Model for 120, 160
        x = np.zeros((2, model.cropped_size[0], model.cropped_size[1], 6))
        for i in range(2):
            x[i, :, :, 0] = frame[i, :, :]
        yield [x.astype(np.float32)]


if __name__ == "__main__":
    try:
        modelfile = input('Enter model filename/location : ').strip(" ")
        print(f"Model filename is {modelfile}")
        # modelname = os.path.split(modelfile)[-1][:-3]
        modelname = modelfile.split(".")[-2]
        modelname2 = os.path.split(modelfile)[-1][:-3]
        print(f"Model name is {modelname}")

        # Convert to onnx format
        model = OFModel(modelfile)
        # Batch changed to 1 to support TPU.
        x = torch.randn(1, 6, model.cropped_size[0], model.cropped_size[1])
        # x = torch.randn(1, 6, 87, 116)
        # x = torch.randn(1, 6, 56, 77)
        torch.onnx.export(model.encoder, x,
                          modelname + ".onnx",
                          export_params=True,
                          input_names=['input'],
                          #   output_names=['output'])
                          output_names=['output1', 'output2'])
        ret_val = 0
    except Exception as e:
        print("Onnx Export Failed: {}\n\n".format(e))
        ret_val = sys.exit(1)

    # Optimise/simplify the onnx model
    if ret_val == 0:
        print("##### Onnx Export Done ##### \n\n")
        ret_val1 = os.system(
            "python3 -m onnxsim {}.onnx {}_opt.onnx".format(modelname, modelname))

    # Model optimizer to convert onnx to openvino
    if ret_val1 == 0:
        print("##### Onnx Simplify Done ##### \n\n")
        # Find your openvino installation and the corresponding path to mo.py
        ret_val2 = os.system(
            "python3 {} --input_model {}_opt.onnx --output_dir {}_openvino/".format(MODELOPTIMIZERPATH, modelname, modelname))

    # Openvino to Tensorflow
    if ret_val2 == 0:
        print("##### Onnx to OpenVino Done ##### \n\n")
        ret_val3 = os.system("openvino2tensorflow \
                  --model_path {}_openvino/{}_opt.xml \
                  --model_output_path {}_saved_model \
                  --output_saved_model".format(modelname, modelname2, modelname))

    # Tensorflow to TF Lite (Incorporates quantisation specific parameters)
    if ret_val3 == 0:
        print("##### OpenVino to TF Done ##### \n\n")
        modelname = modelname + "_saved_model"
        converter = tf.lite.TFLiteConverter.from_saved_model(modelname + "/")
        # Dynamic range quantization (Weights from ftp to int (8bits)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        # Using this mode requires representative datasets.
        converter.representative_dataset = rep_data_gen
        # Explicitly set the ops to allow maxpool function
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                                               tf.lite.OpsSet.SELECT_TF_OPS]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        tflite_model = converter.convert()
        with open('{}.tflite'.format(modelname), 'wb') as f:
            f.write(tflite_model)

    # TF Lite to Edge Compatible model.
        try:
            os.system("edgetpu_compiler {}.tflite".format(modelname))
        except FileNotFoundError:
            print("TF Lite model file not found!")
