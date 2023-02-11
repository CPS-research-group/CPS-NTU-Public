import numpy as np
import torch
import torch.nn as nn
import tflite_runtime.interpreter as tflite

try:
    from pycoral.utils import edgetpu
    from pycoral.adapters import common
except:
    pass

class Detector(nn.Module):

    def __init__(self, modelfile, args):
        super(Detector, self).__init__()

        self.config = {'metric': 0, 'var_horizontal': 1, 'var_vertical': 1}
        self.device = args.device

        if self.device is None:
            # print(modelfile)
            self.interpreter = edgetpu.make_interpreter(modelfile)
            self.interpreter.allocate_tensors()
            self.encoder = Encoder_TPU(args, self.config, self.interpreter)
        
        elif self.device == "TFLITE":
            # Load the TFLite model and allocate tensors.
            self.interpreter = tflite.Interpreter(model_path=modelfile)
            self.interpreter.allocate_tensors()
            self.encoder = Encoder_TFLITE_OF(args, self.config, self.interpreter)

        else:
            # pt modelfile
            self.encoder = Encoder_CPU(args, self.config)
            self.load_state_dict(torch.load(modelfile, map_location=self.device))
            self.eval()
    
    def check(self, data):
        if self.device is None or self.device == "TFLITE":
            # TPU & TFLITE
            # (d_grp1, d_grp2) = self.encoder.encode(data, self.interpreter)
            (d_grp1, d_grp2) = self.encoder.encode(data)
            d_horizontal = np.sum(d_grp1)
            d_vertical = np.sum(d_grp2)
        else:
            # on CPU/GPU Computer
            with torch.no_grad():
                # Testing CPU data format
                data = torch.from_numpy(data).to(self.device)
                (d_grp1, d_grp2) = self.encoder.encode(data)

                d_horizontal = np.sum(d_grp1.cpu().numpy())
                d_vertical = np.sum(d_grp2.cpu().numpy())

        if "th_horizontal" in self.config.keys() and "th_vertical" in self.config.keys():
            if d_horizontal >= self.config["th_horizontal"] \
                    or d_vertical >= self.config["th_vertical"]:
                predict = True
            else:
                predict = False
        else:
            predict = None

        return {"ood": predict, "value_horizontal": d_horizontal, "value_vertical": d_vertical,
                "value_total": (d_horizontal + d_vertical)}

    def check_group(self, data, group):
        if self.device is None or self.device == "TFLITE":
            # TPU & TFLITE
            # (d_grp1, d_grp2) = self.encoder.encode(data, self.interpreter)
            d_grp1 = self.encoder.encode_grp(data, group)
            d = np.sum(d_grp1)
        else:
            # on CPU/GPU Computer
            with torch.no_grad():
                # Testing CPU data format
                data = torch.from_numpy(data).to(self.device)
                (d_grp1) = self.encoder.encode_h(data)
                (d_grp2) = self.encoder.encode_v(data)

                d_horizontal = np.sum(d_grp1.cpu().numpy())
                d_vertical = np.sum(d_grp2.cpu().numpy())
                d = d_horizontal + d_vertical

        return {"value": d}

class Encoder_CPU(nn.Module):

    def __init__(self, args, config):
        super(Encoder_CPU, self).__init__()

        self.device = args.device
        self.input_size = (args.crop_height, args.crop_width)
        self.group, self.n_latents, self.n_frames = 2, 12, args.n_frames
        self.mu1, self.mu2 = 0, 0
        self.metric = config["metric"]
        self.var1 = torch.from_numpy(
            config["var_horizontal"] * np.ones(self.n_latents)).float()
        self.var2 = torch.from_numpy(
            config["var_vertical"] * np.ones(self.n_latents)).float()

        self.var1 = self.var1.to(self.device)
        self.var2 = self.var2.to(self.device)

        self.hiddenunits = 512
        # Input Size: 150,200 -> 143, 188
        if self.input_size[0] == 143:
            print(f"Model input sizes: {self.input_size[0]} x {self.input_size[1]}")
            self.grp1_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp1_conv1_bn = nn.BatchNorm2d(32)
            self.grp1_conv1_ac = nn.ReLU()
            self.grp1_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp1_conv2_bn = nn.BatchNorm2d(64)
            self.grp1_conv2_ac = nn.ReLU()
            self.grp1_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(1, 0), bias=False)
            self.grp1_conv3_bn = nn.BatchNorm2d(128)
            self.grp1_conv3_ac = nn.ReLU()
            self.grp1_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(0, 0), bias=False)
            self.grp1_conv4_bn = nn.BatchNorm2d(256)
            self.grp1_conv4_ac = nn.ReLU()
            self.grp1_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

            self.grp2_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(0, 0),  bias=False)
            self.grp2_conv1_bn = nn.BatchNorm2d(32)
            self.grp2_conv1_ac = nn.ReLU()
            self.grp2_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp2_conv2_bn = nn.BatchNorm2d(64)
            self.grp2_conv2_ac = nn.ReLU()
            self.grp2_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(1, 0), bias=False)
            self.grp2_conv3_bn = nn.BatchNorm2d(128)
            self.grp2_conv3_ac = nn.ReLU()
            self.grp2_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(0, 0), bias=False)
            self.grp2_conv4_bn = nn.BatchNorm2d(256)
            self.grp2_conv4_ac = nn.ReLU()
            self.grp2_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

        # Input Size: 120,160 -> 113,152
        elif self.input_size[0] == 113:
            print(f"Model input sizes: {self.input_size[0]} x {self.input_size[1]}")
            self.grp1_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp1_conv1_bn = nn.BatchNorm2d(32)
            self.grp1_conv1_ac = nn.ReLU()
            self.grp1_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(2, 0), bias=False)
            self.grp1_conv2_bn = nn.BatchNorm2d(64)
            self.grp1_conv2_ac = nn.ReLU()
            self.grp1_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(2, 2), bias=False)
            self.grp1_conv3_bn = nn.BatchNorm2d(128)
            self.grp1_conv3_ac = nn.ReLU()
            self.grp1_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(0, 0), bias=False)
            self.grp1_conv4_bn = nn.BatchNorm2d(256)
            self.grp1_conv4_ac = nn.ReLU()
            self.grp1_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

            self.grp2_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=0,  bias=False)
            self.grp2_conv1_bn = nn.BatchNorm2d(32)
            self.grp2_conv1_ac = nn.ReLU()
            self.grp2_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(2, 0), bias=False)
            self.grp2_conv2_bn = nn.BatchNorm2d(64)
            self.grp2_conv2_ac = nn.ReLU()
            self.grp2_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(2, 2), bias=False)
            self.grp2_conv3_bn = nn.BatchNorm2d(128)
            self.grp2_conv3_ac = nn.ReLU()
            self.grp2_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(0, 0), bias=False)
            self.grp2_conv4_bn = nn.BatchNorm2d(256)
            self.grp2_conv4_ac = nn.ReLU()
            self.grp2_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

        # Input Size: 90,120 -> 87,116
        elif self.input_size[0] == 87:
            print(f"Model input sizes: {self.input_size[0]} x {self.input_size[1]}")
            self.grp1_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(1, 0), bias=False)
            self.grp1_conv1_bn = nn.BatchNorm2d(32)
            self.grp1_conv1_ac = nn.ReLU()
            self.grp1_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp1_conv2_bn = nn.BatchNorm2d(64)
            self.grp1_conv2_ac = nn.ReLU()
            self.grp1_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(1, 1), bias=False)
            self.grp1_conv3_bn = nn.BatchNorm2d(128)
            self.grp1_conv3_ac = nn.ReLU()
            self.grp1_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 1), bias=False)
            self.grp1_conv4_bn = nn.BatchNorm2d(256)
            self.grp1_conv4_ac = nn.ReLU()
            self.grp1_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

            self.grp2_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(1, 0), bias=False)
            self.grp2_conv1_bn = nn.BatchNorm2d(32)
            self.grp2_conv1_ac = nn.ReLU()
            self.grp2_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp2_conv2_bn = nn.BatchNorm2d(64)
            self.grp2_conv2_ac = nn.ReLU()
            self.grp2_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(1, 1), bias=False)
            self.grp2_conv3_bn = nn.BatchNorm2d(128)
            self.grp2_conv3_ac = nn.ReLU()
            self.grp2_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 1), bias=False)
            self.grp2_conv4_bn = nn.BatchNorm2d(256)
            self.grp2_conv4_ac = nn.ReLU()
            self.grp2_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

        # Input Size: 60,80 -> 56,77
        elif self.input_size[0] == 56:
            print(f"Model input sizes: {self.input_size[0]} x {self.input_size[1]}")
            self.grp1_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(3, 2), bias=False)
            self.grp1_conv1_bn = nn.BatchNorm2d(32)
            self.grp1_conv1_ac = nn.ReLU()
            self.grp1_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(3, 0), bias=False)
            self.grp1_conv2_bn = nn.BatchNorm2d(64)
            self.grp1_conv2_ac = nn.ReLU()
            self.grp1_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(2, 3), bias=False)
            self.grp1_conv3_bn = nn.BatchNorm2d(128)
            self.grp1_conv3_ac = nn.ReLU()
            # Kernel 2,2 works
            # self.grp1_conv4 = nn.Conv2d(128, 256, (2,2), stride=(1,1), padding=(0,0), bias=False)
            self.grp1_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 1), bias=False)
            self.grp1_conv4_bn = nn.BatchNorm2d(256)
            self.grp1_conv4_ac = nn.ReLU()
            self.grp1_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

            self.grp2_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(3, 2), bias=False)
            self.grp2_conv1_bn = nn.BatchNorm2d(32)
            self.grp2_conv1_ac = nn.ReLU()
            self.grp2_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(3, 0), bias=False)
            self.grp2_conv2_bn = nn.BatchNorm2d(64)
            self.grp2_conv2_ac = nn.ReLU()
            self.grp2_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(2, 3), bias=False)
            self.grp2_conv3_bn = nn.BatchNorm2d(128)
            self.grp2_conv3_ac = nn.ReLU()
            # Kernel 2,2 works
            # self.grp2_conv4 = nn.Conv2d(128, 256, (2,2), stride=(1,1), padding=(0,0), bias=False)
            self.grp2_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 1), bias=False)
            self.grp2_conv4_bn = nn.BatchNorm2d(256)
            self.grp2_conv4_ac = nn.ReLU()
            self.grp2_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

        # Input Size: 96,128 -> 89,122
        elif self.input_size[0] == 89:
            print(f"Model input sizes: {self.input_size[0]} x {self.input_size[1]}")
            self.grp1_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp1_conv1_bn = nn.BatchNorm2d(32)
            self.grp1_conv1_ac = nn.ReLU()
            self.grp1_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(3, 2), bias=False)
            self.grp1_conv2_bn = nn.BatchNorm2d(64)
            self.grp1_conv2_ac = nn.ReLU()
            self.grp1_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(0, 3), bias=False)
            self.grp1_conv3_bn = nn.BatchNorm2d(128)
            self.grp1_conv3_ac = nn.ReLU()
            self.grp1_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 0), bias=False)
            self.grp1_conv4_bn = nn.BatchNorm2d(256)
            self.grp1_conv4_ac = nn.ReLU()
            self.grp1_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

            self.grp2_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp2_conv1_bn = nn.BatchNorm2d(32)
            self.grp2_conv1_ac = nn.ReLU()
            self.grp2_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(3, 2), bias=False)
            self.grp2_conv2_bn = nn.BatchNorm2d(64)
            self.grp2_conv2_ac = nn.ReLU()
            self.grp2_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(0, 3), bias=False)
            self.grp2_conv3_bn = nn.BatchNorm2d(128)
            self.grp2_conv3_ac = nn.ReLU()
            self.grp2_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 0), bias=False)
            self.grp2_conv4_bn = nn.BatchNorm2d(256)
            self.grp2_conv4_ac = nn.ReLU()
            self.grp2_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

        # Input Size: 72,96 -> 65,86
        elif self.input_size[0] == 65:
            print(f"Model input sizes: {self.input_size[0]} x {self.input_size[1]}")
            self.grp1_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp1_conv1_bn = nn.BatchNorm2d(32)
            self.grp1_conv1_ac = nn.ReLU()
            self.grp1_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(1, 2), bias=False)
            self.grp1_conv2_bn = nn.BatchNorm2d(64)
            self.grp1_conv2_ac = nn.ReLU()
            self.grp1_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(2, 2), bias=False)
            self.grp1_conv3_bn = nn.BatchNorm2d(128)
            self.grp1_conv3_ac = nn.ReLU()
            self.grp1_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 1), bias=False)
            self.grp1_conv4_bn = nn.BatchNorm2d(256)
            self.grp1_conv4_ac = nn.ReLU()
            self.grp1_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

            self.grp2_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp2_conv1_bn = nn.BatchNorm2d(32)
            self.grp2_conv1_ac = nn.ReLU()
            self.grp2_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(1, 2), bias=False)
            self.grp2_conv2_bn = nn.BatchNorm2d(64)
            self.grp2_conv2_ac = nn.ReLU()
            self.grp2_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(2, 2), bias=False)
            self.grp2_conv3_bn = nn.BatchNorm2d(128)
            self.grp2_conv3_ac = nn.ReLU()
            self.grp2_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 1), bias=False)
            self.grp2_conv4_bn = nn.BatchNorm2d(256)
            self.grp2_conv4_ac = nn.ReLU()
            self.grp2_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

        # Input Size: 48,64 -> 41,56
        elif self.input_size[0] == 41:
            print(f"Model input sizes: {self.input_size[0]} x {self.input_size[1]}")
            self.grp1_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp1_conv1_bn = nn.BatchNorm2d(32)
            self.grp1_conv1_ac = nn.ReLU()
            self.grp1_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(2, 1), bias=False)
            self.grp1_conv2_bn = nn.BatchNorm2d(64)
            self.grp1_conv2_ac = nn.ReLU()
            self.grp1_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(3, 4), bias=False)
            self.grp1_conv3_bn = nn.BatchNorm2d(128)
            self.grp1_conv3_ac = nn.ReLU()
            self.grp1_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 1), bias=False)
            self.grp1_conv4_bn = nn.BatchNorm2d(256)
            self.grp1_conv4_ac = nn.ReLU()
            self.grp1_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

            self.grp2_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(0, 0), bias=False)
            self.grp2_conv1_bn = nn.BatchNorm2d(32)
            self.grp2_conv1_ac = nn.ReLU()
            self.grp2_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(2, 1), bias=False)
            self.grp2_conv2_bn = nn.BatchNorm2d(64)
            self.grp2_conv2_ac = nn.ReLU()
            self.grp2_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(3, 4), bias=False)
            self.grp2_conv3_bn = nn.BatchNorm2d(128)
            self.grp2_conv3_ac = nn.ReLU()
            self.grp2_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 1), bias=False)
            self.grp2_conv4_bn = nn.BatchNorm2d(256)
            self.grp2_conv4_ac = nn.ReLU()
            self.grp2_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

        # Input Size: 24,32 -> 21,26
        elif self.input_size[0] == 21:
            print(f"Model input sizes: {self.input_size[0]} x {self.input_size[1]}")
            self.grp1_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(1, 0), bias=False)
            self.grp1_conv1_bn = nn.BatchNorm2d(32)
            self.grp1_conv1_ac = nn.ReLU()
            self.grp1_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(5, 6), bias=False)
            self.grp1_conv2_bn = nn.BatchNorm2d(64)
            self.grp1_conv2_ac = nn.ReLU()
            self.grp1_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(3, 4), bias=False)
            self.grp1_conv3_bn = nn.BatchNorm2d(128)
            self.grp1_conv3_ac = nn.ReLU()
            self.grp1_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 1), bias=False)
            self.grp1_conv4_bn = nn.BatchNorm2d(256)
            self.grp1_conv4_ac = nn.ReLU()
            self.grp1_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

            self.grp2_conv1 = nn.Conv2d(
                self.n_frames, 32, (5, 5), stride=(3, 3), padding=(1, 0), bias=False)
            self.grp2_conv1_bn = nn.BatchNorm2d(32)
            self.grp2_conv1_ac = nn.ReLU()
            self.grp2_conv2 = nn.Conv2d(
                32, 64, (5, 5), stride=(3, 3), padding=(5, 6), bias=False)
            self.grp2_conv2_bn = nn.BatchNorm2d(64)
            self.grp2_conv2_ac = nn.ReLU()
            self.grp2_conv3 = nn.Conv2d(
                64, 128, (5, 5), stride=(3, 3), padding=(3, 4), bias=False)
            self.grp2_conv3_bn = nn.BatchNorm2d(128)
            self.grp2_conv3_ac = nn.ReLU()
            self.grp2_conv4 = nn.Conv2d(
                128, 256, (5, 5), stride=(1, 1), padding=(1, 1), bias=False)
            self.grp2_conv4_bn = nn.BatchNorm2d(256)
            self.grp2_conv4_ac = nn.ReLU()
            self.grp2_linear = nn.Linear(self.hiddenunits, 2*self.n_latents)

    def forward(self, x):
        # print(x.shape)      # [2, 113, 152, 6]
        # X is now [2, 113, 152, 6]
        intermediate = torch.moveaxis(x, -1, 1)         
        output_grp1 = torch.zeros((1, self.n_frames, self.input_size[0], self.input_size[1]))
        output_grp2 = torch.zeros((1, self.n_frames, self.input_size[0], self.input_size[1]))

        output_grp1[:,:,:,:] = intermediate[0,:,:,:]
        output_grp2[:,:,:,:] = intermediate[1,:,:,:]

        output_grp1 = self.grp1_conv1(output_grp1)
        output_grp1 = self.grp1_conv1_bn(output_grp1)
        output_grp1 = self.grp1_conv1_ac(output_grp1)
        output_grp1 = self.grp1_conv2(output_grp1)
        output_grp1 = self.grp1_conv2_bn(output_grp1)
        output_grp1 = self.grp1_conv2_ac(output_grp1)
        output_grp1 = self.grp1_conv3(output_grp1)
        output_grp1 = self.grp1_conv3_bn(output_grp1)
        output_grp1 = self.grp1_conv3_ac(output_grp1)
        output_grp1 = self.grp1_conv4(output_grp1)
        output_grp1 = self.grp1_conv4_bn(output_grp1)
        output_grp1 = self.grp1_conv4_ac(output_grp1)
        output_grp1 = output_grp1.view(output_grp1.size(0), -1)
        output_grp1 = self.grp1_linear(output_grp1)

        output_grp2 = self.grp2_conv1(output_grp2)
        output_grp2 = self.grp2_conv1_bn(output_grp2)
        output_grp2 = self.grp2_conv1_ac(output_grp2)
        output_grp2 = self.grp2_conv2(output_grp2)
        output_grp2 = self.grp2_conv2_bn(output_grp2)
        output_grp2 = self.grp2_conv2_ac(output_grp2)
        output_grp2 = self.grp2_conv3(output_grp2)
        output_grp2 = self.grp2_conv3_bn(output_grp2)
        output_grp2 = self.grp2_conv3_ac(output_grp2)
        output_grp2 = self.grp2_conv4(output_grp2)
        output_grp2 = self.grp2_conv4_bn(output_grp2)
        output_grp2 = self.grp2_conv4_ac(output_grp2)
        output_grp2 = output_grp2.view(output_grp2.size(0), -1)
        output_grp2 = self.grp2_linear(output_grp2)

        # return output_grp1, output_grp2
        return output_grp1.chunk(2, 1), output_grp2.chunk(2, 1)

    def encode(self, input):
        (mu_grp1, logvar_grp1), (mu_grp2, logvar_grp2) = self.forward(input)

        if self.metric == 1:
            # CPU/GPU computation
            priorvar_grp1 = self.var1 * \
                torch.ones(logvar_grp1.size()).to(self.device)
            priorvar_grp2 = self.var2 * \
                torch.ones(logvar_grp2.size()).to(self.device)
            d_grp1 = ((mu_grp1 - self.mu1).pow(2) + logvar_grp1.exp() + priorvar_grp1 -
                      2. * torch.sqrt(logvar_grp1.exp() * priorvar_grp1)).sum(dim=1)

            d_grp2 = ((mu_grp2 - self.mu2) ** 2 + np.exp(logvar_grp2) + priorvar_grp2 -
                      2. * torch.sqrt(np.exp(logvar_grp2) * priorvar_grp2)).sum(dim=1)

        else: # Metric = 0
            pass
            d_grp1 = 0.5 * (mu_grp1.pow(2) + logvar_grp1.exp() - logvar_grp1 - 1).sum(dim=1)
            d_grp2 = 0.5 * (mu_grp2.pow(2) + logvar_grp2.exp() - logvar_grp2 - 1).sum(dim=1)

        return d_grp1, d_grp2

    def encode_h(self, input):
        intermediate = torch.moveaxis(input, -1, 1)         
        output_grp1 = torch.zeros((1, self.n_frames, self.input_size[0], self.input_size[1]))

        output_grp1[:,:,:,:] = intermediate[0,:,:,:]


        output_grp1 = self.grp1_conv1(output_grp1)
        output_grp1 = self.grp1_conv1_bn(output_grp1)
        output_grp1 = self.grp1_conv1_ac(output_grp1)
        output_grp1 = self.grp1_conv2(output_grp1)
        output_grp1 = self.grp1_conv2_bn(output_grp1)
        output_grp1 = self.grp1_conv2_ac(output_grp1)
        output_grp1 = self.grp1_conv3(output_grp1)
        output_grp1 = self.grp1_conv3_bn(output_grp1)
        output_grp1 = self.grp1_conv3_ac(output_grp1)
        output_grp1 = self.grp1_conv4(output_grp1)
        output_grp1 = self.grp1_conv4_bn(output_grp1)
        output_grp1 = self.grp1_conv4_ac(output_grp1)
        output_grp1 = output_grp1.view(output_grp1.size(0), -1)
        output_grp1 = self.grp1_linear(output_grp1)

        (mu_grp1, logvar_grp1) = output_grp1.chunk(2, 1)

        if self.metric == 1:
            # CPU/GPU computation
            priorvar_grp1 = self.var1 * \
                torch.ones(logvar_grp1.size()).to(self.device)
            d_grp1 = ((mu_grp1 - self.mu1).pow(2) + logvar_grp1.exp() + priorvar_grp1 -
                      2. * torch.sqrt(logvar_grp1.exp() * priorvar_grp1)).sum(dim=1)

        else: # Metric = 0
            pass
            d_grp1 = 0.5 * (mu_grp1.pow(2) + logvar_grp1.exp() - logvar_grp1 - 1).sum(dim=1)

        return d_grp1

    def encode_v(self, input):
        intermediate = torch.moveaxis(input, -1, 1)         
        output_grp2 = torch.zeros((1, self.n_frames, self.input_size[0], self.input_size[1]))

        output_grp2[:,:,:,:] = intermediate[0,:,:,:]


        output_grp2 = self.grp2_conv1(output_grp2)
        output_grp2 = self.grp2_conv1_bn(output_grp2)
        output_grp2 = self.grp2_conv1_ac(output_grp2)
        output_grp2 = self.grp2_conv2(output_grp2)
        output_grp2 = self.grp2_conv2_bn(output_grp2)
        output_grp2 = self.grp2_conv2_ac(output_grp2)
        output_grp2 = self.grp2_conv3(output_grp2)
        output_grp2 = self.grp2_conv3_bn(output_grp2)
        output_grp2 = self.grp2_conv3_ac(output_grp2)
        output_grp2 = self.grp2_conv4(output_grp2)
        output_grp2 = self.grp2_conv4_bn(output_grp2)
        output_grp2 = self.grp2_conv4_ac(output_grp2)
        output_grp2 = output_grp2.view(output_grp2.size(0), -1)
        output_grp2 = self.grp2_linear(output_grp2)

        (mu_grp2, logvar_grp2) = output_grp2.chunk(2, 1)

        if self.metric == 1:
            # CPU/GPU computation
            priorvar_grp2 = self.var1 * \
                torch.ones(logvar_grp2.size()).to(self.device)
            d_grp2 = ((mu_grp2 - self.mu1).pow(2) + logvar_grp2.exp() + priorvar_grp2 -
                      2. * torch.sqrt(logvar_grp2.exp() * priorvar_grp2)).sum(dim=1)

        else: # Metric = 0
            pass
            d_grp2 = 0.5 * (mu_grp2.pow(2) + logvar_grp2.exp() - logvar_grp2 - 1).sum(dim=1)

        return d_grp2


class Encoder_TPU(nn.Module):

    def __init__(self, args, config, interpreter):
        super(Encoder_TPU, self).__init__()
        self.input_size = (args.crop_height, args.crop_width)
        self.device = args.device
        self.interpreter = interpreter
        self.group, self.n_latents, self.n_frames = 2, 12, args.n_frames
        # Values from quantization process
        (self.quant_scale, self.quant_zero_point) = self.interpreter.get_input_details()[0]['quantization']
        # Check index for the output accordingly. 
        (self.quant_scale_out1, self.quant_zero_point_out1) = self.interpreter.get_output_details()[1]['quantization']
        (self.quant_scale_out2, self.quant_zero_point_out2) = self.interpreter.get_output_details()[0]['quantization']

        # self.mu1 = config["mu_horizontal"]
        self.mu1, self.mu2 = 0, 0
        self.metric = config["metric"]
        self.var1 = torch.from_numpy(
            config["var_horizontal"] * np.ones(self.n_latents)).float()
        self.var2 = torch.from_numpy(
            config["var_vertical"] * np.ones(self.n_latents)).float()

    def callInterpreter(self, data, interpreter, output_idx):
        common.set_input(interpreter, data)
        interpreter.invoke()
        return interpreter.get_tensor(output_idx)

    def encode(self, flow):
        # Convert real_value flow to int8_value
        flow = (flow / self.quant_scale) + (self.quant_zero_point)        
        # Run on EdgeTPU
        # Input in NHWC format of shape (1, 113, 152, 1)
        # First horizontal frame, take output from branch 0
        x_grp1 = np.zeros((1, self.input_size[0], self.input_size[1], self.n_frames))
        x_grp1 = flow[0, :, :, :]
        output = self.callInterpreter(x_grp1, self.interpreter, 1)
        (mu_grp1, logvar_grp1) = output[0][:self.n_latents], output[0][self.n_latents:]

        # Second frame - vertical frame, take output from branch 1
        x_grp2 = np.zeros((1, self.input_size[0], self.input_size[1], self.n_frames))
        x_grp2 = flow[1, :, :, :]
        output = self.callInterpreter(x_grp2, self.interpreter, 2)
        (mu_grp2, logvar_grp2) = output[0][:self.n_latents], output[0][self.n_latents:]

        # Float to Integer domain (Scaling true prior instead of mu & var values)
        var1 = (self.var1 / self.quant_scale_out1) + (self.quant_zero_point_out1)
        mu1 = (self.mu1 / self.quant_scale_out1) + (self.quant_zero_point_out1)

        if self.metric == 1:    
            mu_grp1, logvar_grp1 = mu_grp1.astype(float), logvar_grp1.astype(float)
            priorvar_grp1 = var1 * np.ones(len(logvar_grp1))
            d_grp1 = [np.power((mu_grp1[i] - mu1), 2) +
                    np.exp(logvar_grp1[i]) for i in range(self.n_latents)]
            d_grp1 = [d_grp1[i] + priorvar_grp1[i] for i in range(self.n_latents)]
            d_grp1 = [d_grp1[i] - 2. *
                    np.sqrt(np.exp(logvar_grp1[i]) * priorvar_grp1[i]) for i in range(self.n_latents)]
            d_grp1 = np.sum(d_grp1)

        # Float to Integer domain (Scaling true prior instead of mu & var values)
        var2 = (self.var2 / self.quant_scale_out2) + (self.quant_zero_point_out2)
        mu2 = (self.mu2 / self.quant_scale_out2) + (self.quant_zero_point_out2)

        if self.metric == 1:     
            mu_grp2, logvar_grp2 = mu_grp2.astype(float), logvar_grp2.astype(float)
            priorvar_grp2 = var2 * np.ones(len(logvar_grp2))
            d_grp2 = [np.power((mu_grp2[i] - mu2), 2) +
                    np.exp(logvar_grp2[i]) for i in range(self.n_latents)]
            d_grp2 = [d_grp2[i] + priorvar_grp2[i] for i in range(self.n_latents)]
            d_grp2 = [d_grp2[i] - 2. *
                    np.sqrt(np.exp(logvar_grp2[i]) * priorvar_grp2[i]) for i in range(self.n_latents)]
            d_grp2 = np.sum(d_grp2)

        else: # Metric 0
            d_grp1 = [0.5 * (np.power((mu_grp1[i] - mu1), 2) + np.exp(logvar_grp1[i] - logvar_grp1[i] - 1)) for i in range(self.n_latents)]
            d_grp1 = np.sum(d_grp1)
            d_grp2 = [0.5 * (np.power((mu_grp2[i] - mu2), 2) + np.exp(logvar_grp2[i] - logvar_grp2[i] - 1)) for i in range(self.n_latents)]
            d_grp2 = np.sum(d_grp2)
        
        return d_grp1, d_grp2

class Encoder_TFLITE_OF(nn.Module):

    def __init__(self, args, config, interpreter):
        super(Encoder_TFLITE_OF, self).__init__()

        self.device = args.device
        self.input_size = (args.crop_height, args.crop_width)
        self.group, self.n_latents, self.n_frames = 2, 12, args.n_frames
        self.interpreter = interpreter
        # NEED TO FOLLOW MODEL VALUES
        self.input_details = self.interpreter.get_input_details()
        # print(self.input_details)
        self.output_details = self.interpreter.get_output_details()
        # print(self.output_details)
        (self.quant_scale, self.quant_zero_point) = self.interpreter.get_input_details()[
            0]['quantization']
        (self.quant_scale_out1, self.quant_zero_point_out1) = interpreter.get_output_details()[
            1]['quantization']
        (self.quant_scale_out2, self.quant_zero_point_out2) = interpreter.get_output_details()[
            0]['quantization']
        print((self.quant_scale_out1, self.quant_zero_point_out1), (self.quant_scale_out2, self.quant_zero_point_out2))
        self.mu1, self.mu2 = 0, 0
        self.metric = config["metric"]
        self.var1 = torch.from_numpy(
            config["var_horizontal"] * np.ones(self.n_latents)).float()
        self.var2 = torch.from_numpy(
            config["var_vertical"] * np.ones(self.n_latents)).float()

    def encode(self, flow):
        # Convert real_value flow to int8_value and cast as int8
        flow = (flow / self.quant_scale) + (self.quant_zero_point)
        flow = flow.astype(np.int8)

        x_grp1 = np.zeros(
            (1, self.input_size[0], self.input_size[1], self.n_frames), dtype=np.int8)
        x_grp1[:,:,:,:] = flow[0, :, :, :]
        # Get input and output tensors.            
        self.interpreter.set_tensor(self.input_details[0]['index'], x_grp1)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[1]['index'])
        (mu_grp1, logvar_grp1) = output_data[0][:self.n_latents], output_data[0][self.n_latents:]

        # Second vertical frame, take output from branch 1
        # x_grp2 = np.zeros((1, 113, 152, self.n_frames))
        x_grp2 = np.zeros(
            (1, self.input_size[0], self.input_size[1], self.n_frames), dtype=np.int8)
        x_grp2[:,:,:,:] = flow[1, :, :, :]
        self.interpreter.set_tensor(self.input_details[0]['index'], x_grp2)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        (mu_grp2, logvar_grp2) = output_data[0][:self.n_latents], output_data[0][self.n_latents:]

        # Change to Integer domain Scaling true prior instead of mu & var values)
        mu1 = (self.mu1 / self.quant_scale_out1) + (self.quant_zero_point_out1)
        mu2 = (self.mu2 / self.quant_scale_out2) + (self.quant_zero_point_out2)

        d_grp1 = [0.5 * (np.power((mu_grp1[i] - mu1), 2) + np.exp(
            logvar_grp1[i] - logvar_grp1[i] - 1)) for i in range(self.n_latents)]
        d_grp1 = np.sum(d_grp1)
        d_grp2 = [0.5 * (np.power((mu_grp2[i] - mu2), 2) + np.exp(
            logvar_grp2[i] - logvar_grp2[i] - 1)) for i in range(self.n_latents)]
        d_grp2 = np.sum(d_grp2)

        return d_grp1, d_grp2

    def encode_grp(self, flow, group):
        # Convert real_value flow to int8_value and cast as int8
        flow = (flow / self.quant_scale) + (self.quant_zero_point)
        flow = flow.astype(np.int8)

        x_grp = np.zeros(
            (1, self.input_size[0], self.input_size[1], self.n_frames), dtype=np.int8)
        x_grp[:,:,:,:] = flow[0, :, :, :]
        # Get input and output tensors.            
        self.interpreter.set_tensor(self.input_details[0]['index'], x_grp)
        self.interpreter.invoke()
        # Group 0 (1), Group 1 (0)
        output_data = self.interpreter.get_tensor(self.output_details[(group+1) % 2]['index'])
        (mu_grp, logvar_grp) = output_data[0][:self.n_latents], output_data[0][self.n_latents:]

        # Change to Integer domain Scaling true prior instead of mu & var values)
        if group == 0:
            mu = (self.mu1 / self.quant_scale_out1) + (self.quant_zero_point_out1)
        else:
            mu = (self.mu2 / self.quant_scale_out2) + (self.quant_zero_point_out2)

        d_grp = [0.5 * (np.power((mu_grp[i] - mu), 2) + np.exp(
            logvar_grp[i] - logvar_grp[i] - 1)) for i in range(self.n_latents)]
        d_grp = np.sum(d_grp)

        return d_grp