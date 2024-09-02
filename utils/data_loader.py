import torchvision
from torch.utils.data import DataLoader
import torch
from  torch.utils.data import random_split

def get_data_loader(
                input_d, 
                n_chan,
                data_path, 
                batch_size,
                split):

    transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_d),
            torchvision.transforms.ToTensor()])

    if n_chan == 1:
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(input_d),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()])


    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=transforms)
    
    len_dataset = len(dataset)
    num_samples = int(split*len_dataset)
    trainset, test = random_split(dataset, [num_samples, len_dataset-num_samples], generator=torch.Generator().manual_seed(42))
    
    len_dataset = len(trainset)
    num_samples = int(split*len_dataset)
    train, val = random_split(trainset, [num_samples, len_dataset- num_samples], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(
        dataset=train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    val_loader = DataLoader(
        dataset=val,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    test_loader = DataLoader(
        dataset=test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True)

    return train_loader, val_loader, test_loader