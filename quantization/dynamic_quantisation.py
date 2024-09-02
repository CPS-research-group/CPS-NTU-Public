import torch

def dynamic_quantise(model):

    backend = 'qnnpack' # running on on ARM. Use "fbgemm" if running on x86 CPU
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
    )   
    return quantized_model
