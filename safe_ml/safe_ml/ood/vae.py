import argparse
import math
import os
import time
import torch
import torchvision
import pytorch_lightning
import numpy
import cv2

from PIL import Image

class EncConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, k_conv, k_pool, activation):
        super().__init__()
        #print(f'enc_in: {in_channels}')
        #print(f'enc_out: {out_channels}')
        self.conv = torch.nn.Conv2d(in_channels, out_channels, k_conv, padding=k_conv // 2)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.activation = activation
        self.pool = torch.nn.MaxPool2d(k_pool, return_indices=True, ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return self.pool(x)

class DecConvBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, k_conv, k_pool, activation):
        super().__init__()
        #print(f'dec_in: {in_channels}')
        #print(f'dec_out: {out_channels}')
        self.unpool = torch.nn.MaxUnpool2d(k_pool)
        self.conv_t = torch.nn.ConvTranspose2d(in_channels, out_channels, k_conv, padding=k_conv // 2)
        self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        self.activation = activation

    def forward(self, x, indices, out_size):
        x = self.unpool(x, indices, output_size=out_size)
        x = self.conv_t(x)
        x = self.batch_norm(x)
        return self.activation(x)

class FcLayer(torch.nn.Module):
    
    def __init__(self, in_neurons, out_neurons, activation):
        super().__init__()
        #print(f'in: {in_neurons}')
        #print(f'out: {out_neurons}')
        self.linear = torch.nn.Linear(in_neurons, out_neurons)
        self.activation = activation

    def forward(self, x):
        x = self.linear(x)
        return self.activation(x)

class CnnEncoder(torch.nn.Module):

    def __init__(self, conv_blocks, fc_layers):
        super().__init__()
        self.conv_blocks = torch.nn.ModuleList([EncConvBlock(**c) for c in conv_blocks])
        self.fc_layers = torch.nn.ModuleList([FcLayer(**f) for f in fc_layers])

    def forward(self, x):
        indices_l = []
        out_sizes = []
        for block in self.conv_blocks:
            out_sizes.append(x.shape)
            x, indices = block(x)
            #print(x.shape)
            indices_l.append(indices)
        out_sizes.append(x.shape)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)
        return x[:, :x.shape[-1] // 2], x[:, x.shape[-1] // 2:], indices_l, out_sizes

class CnnDecoder(torch.nn.Module):

    def __init__(self, fc_layers, conv_blocks):
        super().__init__()
        self.fc_layers = torch.nn.ModuleList([FcLayer(**f) for f in fc_layers])
        self.conv_blocks = torch.nn.ModuleList([DecConvBlock(**c) for c in conv_blocks])

    def forward(self, x, indices_l, out_sizes):
        for layer in self.fc_layers:
            #print(x.shape)
            x = layer(x)
        x = torch.reshape(x, out_sizes.pop())
        for block in self.conv_blocks:
            x = block(x, indices_l.pop(), out_sizes.pop())
        return x
        

class Vae(pytorch_lightning.LightningModule):

    def __init__(self, conv_blocks, fc_layers, beta=1, learning_rate=1e-5):
        super().__init__()
        self.beta = beta
        self.learning_rate = learning_rate
        self.n_latent = fc_layers[-1]['out_neurons'] // 2
        dec_conv_blocks = []
        dec_fc_layers = [ 
            {
                'in_neurons': fc_layers[-1]['out_neurons'] // 2,
                'out_neurons': fc_layers[-1]['in_neurons'],
                'activation': fc_layers[-1]['activation'],
            }
        ]
        for layer in fc_layers[-2::-1]:
            dec_fc_layers.append(
                {
                    'in_neurons': layer['out_neurons'],
                    'out_neurons': layer['in_neurons'],
                    'activation': layer['activation'],
                }
            )
        for block in conv_blocks[::-1]:
            dec_conv_blocks.append(
                {
                    'in_channels': block['out_channels'],
                    'out_channels': block['in_channels'],
                    'k_conv': block['k_conv'],
                    'k_pool': block['k_pool'],
                    'activation': block['activation']
                }
            )
        dec_conv_blocks[-1]['activation'] = torch.nn.Sigmoid()
        self.encoder = CnnEncoder(conv_blocks, fc_layers)
        self.decoder = CnnDecoder(dec_fc_layers, dec_conv_blocks)


    def forward(self, x):
        mu, logvar, indices, out_sizes = self.encoder(x)
        stdev = torch.exp(logvar / 2)
        eps = torch.randn_like(stdev)
        z = mu + stdev * eps
        return self.decoder(z, indices, out_sizes), mu, logvar

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        x_hat, mu, logvar = self.forward(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
        loss = mse_loss + self.beta * kl_loss
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_hat, mu, logvar = self.forward(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        mse_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')
        loss = mse_loss + self.beta * kl_loss
        self.log('val_loss', loss)
        self.log('y', y)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        _, mu, logvar = self.forward(x)
        kl_loss = 0.5 * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
        self.log('dkl', kl_loss)
        return kl_loss, y


    def predict_step(self, predict_batch, batch_idx):
        x, y = predict_batch
        start = time.time()
        x_hat, mu, logvar = self.forward(x)
        return x_hat, mu, logvar, y, time.time() - start


class OodDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, train_path, val_path, height, width, batch_size=1):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        train_set = torchvision.datasets.ImageFolder(
            root=self.hparams.train_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((self.hparams.height, self.hparams.width))
            ])
        )
        return torch.utils.data.DataLoader(
            train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)

    def val_dataloader(self):
        val_set = torchvision.datasets.ImageFolder(
            root=self.hparams.val_path,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((self.hparams.height, self.hparams.width))
            ])
        )
        return torch.utils.data.DataLoader(
            val_set,
            batch_size=min(len(val_set), self.hparams.batch_size),
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                root=self.hparams.val_path,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize((self.hparams.height, self.hparams.width))
                ])
            ),
            batch_size=self.hparams.batch_size,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(
                root=self.hparams.val_path,
                transform=torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Resize((self.hparams.height, self.hparams.width))
                ])
            ),
            batch_size=self.hparams.batch_size,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True)


class OpticalFlowDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, train_path, val_path, height, width, flows, orientation, batch_size=1):
        super().__init__()
        self.save_hyperparameters()

    def train_dataloader(self):
        train_set = OpticalFlowDataset(
            root=self.hparams.train_path,
            size=(self.hparams.height, self.hparams.width),
            flow_depth=self.hparams.flows,
            orientation=self.hparams.orientation,
        )
        return torch.utils.data.DataLoader(
            train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True
        )

    def val_dataloader(self):
        val_set = OpticalFlowDataset(
            root=self.hparams.val_path,
            size=(self.hparams.height, self.hparams.width),
            flow_depth=self.hparams.flows,
            orientation=self.hparams.orientation,
        )
        return torch.utils.data.DataLoader(
            val_set,
            batch_size=min(len(val_set), self.hparams.batch_size),
            num_workers=len(os.sched_getaffinity(0)),
            drop_last=True
        )


class OpticalFlowDataset(torch.utils.data.Dataset):

    def __init__(self, root: str, size, flow_depth=1, orientation='horiz'):
        super().__init__()
        self.root = root
        self.data = []
        flow_buffer = numpy.zeros((flow_depth, size[0], size[1]))
        flow_buffer_ptr = 0
        flow = None
        last_image = None
        for name in os.listdir(root):
            curr_image = cv2.imread(os.path.join(root, name), cv2.IMREAD_GRAYSCALE)
            curr_image = cv2.resize(curr_image, (size[1], size[0]))
            if last_image is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    last_image,
                    curr_image,
                    flow,
                    pyr_scale=0.5,
                    levels=1,
                    iterations=1,
                    winsize=15,
                    poly_n=5,
                    poly_sigma=1.1,
                    flags=0 if flow is None else cv2.OPTFLOW_USE_INITIAL_FLOW
                )
                flow_buffer[flow_buffer_ptr % flow_depth, :, :] = numpy.copy(flow[:, :, 0 if orientation == 'horiz' else 1])
                if flow_buffer_ptr > flow_depth:
                    self.data.append(flow_buffer)
                flow_buffer_ptr += 1
            last_image = numpy.copy(curr_image)
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return (torch.from_numpy(self.data[idx]).float(), 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a VAE')
    parser.add_argument(
        '--dimensions',
        help='Dimensions of the input image.  format: height x width'
    )
    parser.add_argument(
        '--train_dataset',
        help='Path to folder of training images'
    )
    parser.add_argument(
        '--val_dataset',
        help='Path to folder of validation images'
    )
    conv_blocks = []
    args = parser.parse_args()
    dimensions = max([int(d) for d in args.dimensions.split('x')])
    channels = 3
    while dimensions > 1:
        conv_blocks.append(
            {
                'in_channels': channels,
                'out_channels': 2 ** math.ceil(math.log2(channels) + 1),
                'k_conv': 3,
                'k_pool': 2,
                'activation': torch.nn.LeakyReLU(),
            }
        )
        dimensions //= 2
        channels = 2 ** math.ceil(math.log2(channels) + 1)
    conv_blocks.append(
        {
            'in_channels': channels,
            'out_channels': 2 ** math.ceil(math.log2(channels) + 1),
            'k_conv': 3,
            'k_pool': 2,
            'activation': torch.nn.LeakyReLU(),
        }
    )
    channels = 2 ** math.ceil(math.log2(channels) + 1)
    fc_blocks = [
        {
            'in_neurons': channels,
            'out_neurons': 1024,
            'activation': torch.nn.LeakyReLU()
        },
        {
            'in_neurons': 1024,
            'out_neurons': 64,
            'activation': torch.nn.Identity()
        }
    ]
    model = Vae(conv_blocks, fc_blocks)
    height, width = tuple([int(d) for d in args.dimensions.split('x')])
    data = OodDataModule(args.train_dataset, args.val_dataset, height, width, batch_size=32) 
    trainer = pytorch_lightning.Trainer(
        accelerator='gpu',
        devices=1,
        #auto_scale_batch_size='power',
        #auto_lr_find=True,
        deterministic=True,
        min_epochs=50,
        max_epochs=500,
        log_every_n_steps=1,
        logger=pytorch_lightning.loggers.TensorBoardLogger(save_dir='logs/'),
        callbacks=[
            pytorch_lightning.callbacks.early_stopping.EarlyStopping(
                monitor='val_loss',
                verbose=True,
            )
        ]
    )
    trainer.tune(model, datamodule=data)
    trainer.fit(model, datamodule=data)
    torch.save(model, f'vae{args.dimensions}.pt')

