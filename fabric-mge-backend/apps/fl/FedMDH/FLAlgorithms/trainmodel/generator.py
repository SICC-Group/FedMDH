import torch
import torch.nn as nn
import torch.nn.functional as F
MAXLOG = 0.1
from torch.autograd import Variable
import collections
import numpy as np
from apps.fl.FedMDH.utils.model_config import GENERATORCONFIGS


class Generator(nn.Module):
    def __init__(self, dataset='material', embedding=False, latent_layer_idx=0):
        super(Generator, self).__init__()
        #print("Dataset {}".format(dataset))
        self.embedding = embedding
        self.latent_layer_idx = latent_layer_idx
        self.hidden_dim, self.latent_dim, self.input_channel, self.noise_dim = GENERATORCONFIGS[dataset]
        input_dim = self.noise_dim * 2 if self.embedding else self.noise_dim +1  # Remove label-related input
        self.fc_configs = [input_dim, self.hidden_dim]
        self.init_loss_fn()
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def init_loss_fn(self):
        self.dist_loss = nn.MSELoss()
        self.diversity_loss = DiversityLoss(metric='l1')

    def build_network(self):
        ### FC modules ####
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            #print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim)
            act = nn.ReLU()
            self.fc_layers += [fc, bn, act]
        ### Representation layer
        self.representation_layer = nn.Linear(self.fc_configs[-1], self.latent_dim)
        #print("Build last layer {} X {}".format(self.fc_configs[-1], self.latent_dim))

    def forward(self, labels, latent_layer_idx=0, prior=None, verbose=True):

        result = {}
        batch_size = labels.shape[0]

        # Generate Gaussian noise if needed (still common in regression tasks, but optional)
        if prior==None:
           eps = torch.rand((batch_size, self.noise_dim))  # Sampling from Gaussian
        else:
           eps = prior.sampling_gaussian(batch_size, prior.mu, prior.logvar)
           
        # Adjust eps shape to (self.noise_dim, batch_size)
        #eps = eps.permute(1, 0)  # Swap dimensions

        

         # For regression tasks, use the labels directly as input
        y_input = labels.float().view(batch_size, -1)  # Ensure labels are float and have correct shape
        #y_input=labels.view(32, 1)
         # Concatenate noise and labels (if using noise)
        z = torch.cat((eps, y_input), dim=1)  # Concatenate with original dimensions before fully connected layers

        # Pass through fully connected layers
        for layer in self.fc_layers:
            #print(layer)
            z = layer(z)

        # Generate the final representation or output
        z = self.representation_layer(z)
        # eps= self.representation_layer(eps)
        if verbose:
           result['eps'] = eps
        # Adjust the output shape to be (self.noise_dim, batch_size)
        result['output'] = z  # Swap dimensions
        return result

    @staticmethod
    def normalize_images(layer):
        """
        Normalize images into zero-mean and unit-variance.
        """
        mean = layer.mean(dim=(2, 3), keepdim=True)
        std = layer.view((layer.size(0), layer.size(1), -1)) \
            .std(dim=2, keepdim=True).unsqueeze(3)
        return (layer - mean) / std
    

# 判别器类
class Discriminator(nn.Module):
    def __init__(self, dataset='material', embedding=False):
        super(Discriminator, self).__init__()
        #print("Dataset {}".format(dataset))
        self.embedding = embedding
        self.hidden_dim, self.latent_dim, self.input_channel, self.noise_dim = GENERATORCONFIGS[dataset]
        input_dim = self.noise_dim +1 if not self.embedding else self.noise_dim * 2 + 1  # +1 for label, if not using embedding
        self.fc_configs = [input_dim, self.hidden_dim, 1]  # The final output is a single value (real or fake)
        self.build_network()

    def get_number_of_parameters(self):
        pytorch_total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return pytorch_total_params

    def build_network(self):
        ### FC modules ####
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.fc_configs) - 1):
            input_dim, out_dim = self.fc_configs[i], self.fc_configs[i + 1]
            #print("Build layer {} X {}".format(input_dim, out_dim))
            fc = nn.Linear(input_dim, out_dim)
            bn = nn.BatchNorm1d(out_dim) if i < len(self.fc_configs) - 2 else None
            act = nn.LeakyReLU(0.2) if i < len(self.fc_configs) - 2 else None
            layers = [fc]
            if bn:
                layers.append(bn)
            if act:
                layers.append(act)
            self.fc_layers += layers

    def forward(self, data, labels):
        """
        D(X|y):
        Discriminate between real and fake data conditional on labels.
        :param data: Input data, either real or generated.
        :param labels: Corresponding labels for the data.
        :return: Discriminator output, a value indicating real or fake.
        """
        batch_size = data.shape[0]

        # Ensure labels are float and have correct shape
        y_input = labels.float().view(data.shape[0], -1)
        
        # Concatenate data and labels
        d_input = torch.cat((data, y_input), dim=1)
        

        # Pass through fully connected layers
        for layer in self.fc_layers:
            d_input = layer(d_input)

        # Final output is a single value per data point (real or fake score)
        return d_input


class DivLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """

    def __init__(self):
        """
        Class initializer.
        """
        super().__init__()

    def forward2(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        chunk_size = layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer=layer.view((layer.size(0), -1))
        chunk_size=layer.size(0) // 2

        ####### diversity loss ########
        eps1, eps2=torch.split(noises, chunk_size, dim=0)
        chunk1, chunk2=torch.split(layer, chunk_size, dim=0)
        lz=torch.mean(torch.abs(chunk1 - chunk2)) / torch.mean(
            torch.abs(eps1 - eps2))
        eps=1 * 1e-5
        diversity_loss=1 / (lz + eps)
        return diversity_loss


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))
