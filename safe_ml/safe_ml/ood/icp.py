#/usr/bin/env python3
import torch

class Icp(torch.nn.Module):

    def __init__(self, calibration_set: torch.Tensor) -> None:
        super(Icp, self).__init__()
        calibration_set, _ = torch.sort(calibration_set)
        self.weight = torch.nn.Parameter(calibration_set, requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.sum(torch.where(x < self.weight, 0, 1)) + 1) / (self.weight.size()[0] + 1)

    def extra_repr(self) -> str:
        return 'calibration_set_size={}'.format(self.weight.size)
