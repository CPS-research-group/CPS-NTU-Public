''' 
Model conversion for BVAE Models
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

# Optical Flow Model for TPU
from bvae import Encoder

MODELOPTIMIZERPATH = "/opt/intel/openvino_2021/deployment_tools/model_optimizer/mo.py"


class BvaeModel(nn.Module):

    def __init__(self, modelfile):
        super(BvaeModel, self).__init__()

        self.device = torch.device("cpu")
        self.n_chan = 1 if 'bw' in modelfile else 3
        print(f"Channels: {self.n_chan}")
        self.in_pxls = [int(i) for i in modelfile.replace(
            '.pt', '').split('_')[-1].split('x')]
        print(f"Pixels: {self.in_pxls}")
        self.encoder = Encoder(
            n_latent=36,
            n_chan=self.n_chan,
            input_d=self.in_pxls)
        print("Encoder initialised")
        encoder_sd = torch.load(modelfile)
        self.encoder.load_state_dict(encoder_sd)
        print("State_Dict loaded")
        self.encoder.eval()
        print("Model intialised!")


def rep_data_gen(imagefile="../typical_datasets/typicalset_duckietown_224x224_train_id.npy"):
    # [Images, C, H, W]
    image_array = np.load(imagefile, allow_pickle=True)
    for frame in image_array:
        # print(frame.shape)    # (3, 224, 224)
        frame = np.transpose(frame, (1, 2, 0))  # Convert to HWC
        frame = cv2.resize(frame, (model.in_pxls[1], model.in_pxls[0])) # Resize to model input size
        # frame = frame[:, model.in_pxls, model.in_pxls]
        
        # Original (1, 224, 224, 3)
        x = np.zeros((1, model.in_pxls[0], model.in_pxls[1], model.n_chan))
        x[0, :, :, :] = frame[:, :, :]
        yield [x.astype(np.float32)]


if __name__ == "__main__":
    try:
        modelfile = input('Enter model filename/location : ').strip(" ")
        print(modelfile)
        # modelname = os.path.split(modelfile)[-1][:-3]
        modelname = modelfile.split(".")[-3] + "." + modelfile.split(".")[-2]
        modelname2 = os.path.split(modelfile)[-1][:-3]
        print(modelname)

        # convert to onnx format
        model = BvaeModel(modelfile)
        # Batch changed to 2 to duplicate one sample (OF)
        x = torch.randn(1, model.n_chan, model.in_pxls[0], model.in_pxls[1])
        torch.onnx.export(model.encoder, x,
                          modelname + ".onnx",
                          export_params=True,
                          input_names=['input'],
                          output_names=['output'])
        #                  output_names=['output1', 'output2'])
        ret_val = 0
    except Exception as e:
        print("Onnx Export Failed: \n\n{}".format(e))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        ret_val = sys.exit(1)

    # optimise/simplify the onnx model
    if ret_val == 0:
        print("##### Onnx Export Done ##### \n\n")
        ret_val1 = os.system(
            "python3 -m onnxsim {}.onnx {}_opt.onnx".format(modelname, modelname))

    # Model optimizer to convert onnx to openvino
    if ret_val1 == 0:
        print("##### Onnx Simplify Done ##### \n\n")
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
