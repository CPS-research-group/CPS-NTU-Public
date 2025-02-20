from Parameters import *
import torch
import torchvision.transforms as tran


# state transition function
resize_to_tensor = tran.Compose([tran.ToPILImage(),
                                 tran.Resize(STATE_SIZE),
                                 tran.ToTensor()])


def trans_state(obs):
    return 255 * resize_to_tensor(obs).type(torch.float32).unsqueeze(0).to(RUNNER_DEVICE)


# env joint for multi-agents
class Joint:
    def __init__(self):
        self.action = {}
        raise NotImplementedError