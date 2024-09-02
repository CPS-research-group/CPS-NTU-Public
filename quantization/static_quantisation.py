import torch
from utils.data_loader import get_data_loader
from constants import *

# Static quantization of a model consists of the following steps:
#     Fuse modules
#     Insert Quant/DeQuant Stubs
#     Prepare the fused module (insert observers before and after layers)
#     Calibrate the prepared module (pass it representative data)
#     Convert the calibrated module (replace with quantized version)


def static_quantise(model, calibration=False):

    backend = 'qnnpack' # running on on ARM. Use "fbgemm" if running on x86 CPU
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    
    torch.quantization.prepare(model, inplace=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if calibration:
        with torch.inference_mode():
            
            _, val_loader, _ = get_data_loader(INPUT_DIMENSIONS, N_CHANNELS, TRAIN_DATAPATH, BATCH, split=0.8)

            num_batches = len(val_loader)
            for i, data in enumerate(val_loader):    
                input, _ = data
                input = input.to(device)
                model(input)
                print("Calibrating: [{}/{}]".format(i, num_batches), end='\r')
    
    q_model = torch.quantization.convert(model, inplace=False)
    return q_model

