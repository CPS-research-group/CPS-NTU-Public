#!/bin/bash

# Original Model: leakyReLU learning logvar
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 bvae.py \
  --beta 2.32 \
  --n_latent 36 \
  --dimensions 224x224 \
  --dataset ../data/train_bvae/ \
  --batch 64 \
  --activation leaky \
  --head2learnwhat logvar \
  --epochs 100 \
  --interpolation bilinear \
  train
mv bvae_n36_b2.32__224x224.pt bvae_n36_b2.32__224x224_leaky_logvar.pt
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 bvae.py \
  --weights bvae_n36_b2.32__224x224_leaky_logvar.pt \
  --dataset ../data/train_bvae/ \
  mig

# ReLU learning logvar
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 bvae.py \
  --beta 2.32 \
  --n_latent 36 \
  --dimensions 224x224 \
  --dataset ../train/train_bvae/ \
  --batch 64 \
  --activation relu \
  --head2learnwhat logvar \
  --epochs 100 \
  --interpolation bilinear \
  train
mv bvae_n36_b2.32__224x224.pt bvae_n36_b2.32__224x224_relu_logvar.pt
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 bvae.py \
  --weights bvae_n36_b2.32__224x224_relu_logvar.pt \
  --dataset ../data/train_bvae/ \
  mig

# ReLU learning negative logvar
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 bvae.py \
  --beta 2.32 \
  --n_latent 36 \
  --dimensions 224x224 \
  --dataset ../train/train_bvae/ \
  --batch 64 \
  --activation relu \
  --head2learnwhat neglogvar \
  --epochs 100 \
  --interpolation bilinear \
  train
mv bvae_n36_b2.32__224x224.pt bvae_n36_b2.32__224x224_relu_neglogvar.pt
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 bvae.py \
  --weights bvae_n36_b2.32__224x224_relu_neglogvar.pt
  --dataset ../data/train_bvae/ \
  mig

# ReLU learning variance
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 bvae.py \
  --beta 2.32 \
  --n_latent 36 \
  --dimensions 224x224 \
  --dataset ../train/train_bvae/ \
  --batch 64 \
  --activation relu \
  --head2learnwhat var \
  --epochs 100 \
  --interpolation bilinear \
  train
mv bvae_n36_b2.32__224x224.pt bvae_n36_b2.32__224x224_relu_var.pt
CUBLAS_WORKSPACE_CONFIG=:4096:8 python3 bvae.py \
  --weights bvae_n36_b2.32__224x224_relu_var.pt
  --dataset ../data/train_bvae/ \
  mig
