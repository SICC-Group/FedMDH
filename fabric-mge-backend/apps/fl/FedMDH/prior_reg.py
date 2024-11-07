#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

# Regularizer used in proximal-relational autoencoder
class Prior(nn.Module):
    def __init__(self, z_dim, mean_var=10):
        super(Prior, self).__init__()
        self.z_dim = z_dim
        self.mu_init = mean_var
        self.mu = self.mu_init.clone()
        #self.logvar = torch.ones(data_size)
        self.logvar = torch.zeros(z_dim)
        self.mu_temp = torch.zeros(z_dim)
        self.n_update = 0
        self.mu_local = None

    def forward(self):
        return self.mu, self.logvar

    def sampling_gaussian(self, num_sample, mean, logvar):
        num_sample = num_sample # =batch_size
        self.z_dim = mean.shape # =dim_latent
        std = torch.exp(0.5 * logvar)
        samples = mean + torch.randn(num_sample, self.z_dim[0]) * std
        return samples
    
    # def sampling_gmm(self,num_sample):
    #     std = torch.exp(0.5 * self.logvar)
    #     n = int(num_sample / self.mu.size(0)) + 1
    #     for i in range(n):
    #         eps = torch.randn_like(std)
    #         if i == 0:
    #             samples = self.mu + eps * std
    #         else:
    #             samples = torch.cat((samples, self.mu + eps * std), dim=0)
    #     return samples[:num_sample, :]
    
    def sampling_gmm(self, num_sample):
    # 计算标准差
        std = torch.exp(0.5 * self.logvar)
        # 从高斯分布中采样
        eps = torch.randn((num_sample, self.z_dim))
        samples = self.mu.unsqueeze(0).repeat(num_sample, 1) + eps * std.unsqueeze(0)
        return samples
    
    def init_mu_temp(self):
        self.mu_temp = torch.zeros(self.z_dim)
        self.n_update = 0


# Simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


