import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from torch.utils.data import DataLoader
import numpy as np
import copy
from apps.fl.FedMDH.FLAlgorithms.trainmodel.RKLDivLoss import RKLDivLoss
from apps.fl.FedMDH.utils.model_utils import get_dataset_name
#from torch.nn.modules.loss import RKLDivLoss
from apps.fl.FedMDH.utils.model_config import RUNCONFIGS
from apps.fl.FedMDH.FLAlgorithms.optimizers.fedoptimizer import pFedIBOptimizer
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

class User:
    """
    Base class for users in federated learning.
    """
    def __init__(
            self, args, id, model, train_data, test_data, use_adam=False):
        self.model = copy.deepcopy(model)#需要对MLP的模型进行改进，0代表模型，1代表模型名称
        #self.latent_model=copy.deepcopy(latent_model)
        #self.model_name = model[1]
        self.id = id  # integer
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.dim_latent=args.dim_latent
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.beta = args.beta
        self.lamda = args.lamda
        self.local_epochs = args.local_epochs
        self.algorithm = args.algorithm
        self.K = args.K
        self.dataset = args.dataset
        self.train_data = train_data
        self.test_data = test_data
        #self.trainloader = DataLoader(train_data, self.batch_size, drop_last=False)
        self.trainloader = DataLoader(train_data, self.batch_size, shuffle=True, drop_last=True)
        self.testloader =  DataLoader(test_data, self.batch_size, drop_last=False)
        self.testloaderfull = DataLoader(test_data, self.test_samples)
        self.trainloaderfull = DataLoader(train_data, self.train_samples)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)
        dataset_name = get_dataset_name(self.dataset)
        self.generative_alpha = RUNCONFIGS[dataset_name]['generative_alpha']
        self.generative_beta = RUNCONFIGS[dataset_name]['generative_beta']

        self.local_model = copy.deepcopy(list(self.model.parameters()))
        self.personalized_model_bar = copy.deepcopy(list(self.model.parameters()))
        self.prior_decoder = None
        self.prior_params = None

        self.init_loss_fn()
        if use_adam:
            self.optimizer=torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.learning_rate, betas=(0.9, 0.999),
                eps=1e-08, weight_decay=1e-2, amsgrad=False)
        else:
            self.optimizer = pFedIBOptimizer(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.label_counts = {}#改




    def init_loss_fn(self):
        self.loss=nn.MSELoss()
        self.ensemble_loss=RKLDivLoss()

    def set_parameters(self, model,beta=1):
        for old_param, new_param, local_param in zip(self.model.parameters(), model.parameters(), self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = beta * new_param.data.clone() + (1 - beta)  * old_param.data.clone()
                local_param.data = beta * new_param.data.clone() + (1-beta) * local_param.data.clone()

    def set_prior_decoder(self, model, beta=1):
        for new_param, local_param in zip(model.personal_layers, self.prior_decoder):
            if beta == 1:
                local_param.data = new_param.data.clone()
            else:
                local_param.data = beta * new_param.data.clone() + (1 - beta) * local_param.data.clone()


    def set_prior(self, model):
        for new_param, local_param in zip(model.get_encoder() + model.get_decoder(), self.prior_params):
            local_param.data = new_param.data.clone()

    def set_mask(self, mask_model):
        for new_param, local_param in zip(mask_model.get_masks(), self.mask_model.get_masks()):
            local_param.data = new_param.data.clone()

    def set_shared_parameters(self, model, mode='decode'):
        # only copy shared parameters to local
        for old_param, new_param in zip(
                self.model.get_parameters_by_keyword(mode),
                model.get_parameters_by_keyword(mode)
        ):
            old_param.data = new_param.data.clone()

    def get_parameters(self):
        for param in self.model.parameters():
            param.detach()
        return self.model.parameters()


    def clone_model_paramenter(self, param, clone_param):
        with torch.no_grad():
            for param, clone_param in zip(param, clone_param):
                clone_param.data = param.data.clone()
        return clone_param
    
    def get_updated_parameters(self):
        return self.local_weight_updated
    
    def update_parameters(self, new_params, keyword='all'):
        for param , new_param in zip(self.model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_grads(self):
        grads = []
        for param in self.model.parameters():
            if param.grad is None:
                grads.append(torch.zeros_like(param.data))
            else:
                grads.append(param.grad.data)
        return grads

    def select_top_features(self,x_train: torch.Tensor, y_train: torch.Tensor, num_features: int) -> torch.Tensor:
    # 确保 x_train 和 y_train 都在 CPU 上并且是浮点数
        x_train = x_train.float().cpu()
        y_train = y_train.float().cpu()

    # 计算每个特征与目标的相关性
        x_mean = torch.mean(x_train, dim=0)
        y_mean = torch.mean(y_train)

    # 计算 x 和 y 的协方差
        covariance_xy = torch.mean((x_train - x_mean) * (y_train.unsqueeze(1) - y_mean), dim=0)

        # 计算 x 的方差和 y 的方差
        variance_x = torch.mean((x_train - x_mean) ** 2, dim=0)
        variance_y = torch.mean((y_train - y_mean) ** 2)

        # 计算相关系数（Pearson相关系数）
        correlation_coefficients = covariance_xy / torch.sqrt(variance_x * variance_y)

        # 选择相关性最高的特征
        target_correlations = torch.abs(correlation_coefficients)  # 获取绝对值
        top_features = torch.argsort(target_correlations, descending=True)[:num_features]  # 选择相关性最高的特征
        top_features, _ = torch.sort(top_features)

        return top_features
    
    def test(self,net_glob=None,net_preproc=None,n=0):
        self.model.eval()
        total_loss = 0
        total_mse = 0  # 用于存储累计的 MSE
        total_r2=0
        total_rmse=0
        total_mae=0
        num_samples = 0
        for x, y in self.testloaderfull:
            x = x.float()#新加入的修改bug
            if net_preproc is not None:
                top_features = self.select_top_features(x, y, n)
                print("test_top_features",top_features)
                # 提取最相关的特征，赋值给 x_train1
                x1 = x[:, top_features]
                x_pre = net_preproc(x1)
                output = net_glob(x_pre)['output'].squeeze(-1)
            else:
                 x1= self.select_top_features(x, y,n)
                 output = self.model(x1)['output'].squeeze(-1)
            loss = self.loss(output, y)  # 计算损失
            total_loss += loss.item() * y.size(0)  # 将损失乘以样本数量，方便后续计算平均损失

            # 计算 MSE 并累加
            mse = torch.mean((output - y) ** 2).item()  # 计算当前批次的 MSE
            total_mse += mse * y.size(0)  # 累加每个样本的 MSE
            mae= torch.mean(torch.abs(output - y)).item()
            total_mae +=mae * y.size(0)
            rmse= torch.sqrt(torch.tensor(mse))
            total_rmse += rmse * y.size(0)  # 累加每个样本的 MSE
            r2=r2_score(y.detach().numpy(), output.detach().numpy())
            total_r2 +=r2 * y.size(0)
            num_samples += y.size(0)  # 统计样本总数

        average_loss = total_loss / num_samples  # 计算平均损失
        average_mse = total_mse / num_samples  # 计算平均 MSE
        average_mae= total_mae / num_samples
        average_rmse=total_rmse/num_samples
        average_r2 = total_r2 / num_samples
        return average_mse, average_loss, average_mae,average_rmse,average_r2,num_samples,output,y


    def test_personalized_model(self,net_glob=None,net_preproc=None,n=0):
        self.model.eval()
        test_acc = 0
        loss = 0
        test_mse=0
        self.update_parameters(self.personalized_model_bar)
        for x, y in self.testloaderfull:
            if net_preproc is not None:
                pca = PCA(n_components=n)
                X_pcam = pca.fit_transform(x)
                X_pcam = torch.tensor(X_pcam, dtype=torch.float32)
                X_pca = net_preproc(X_pcam)
                output = net_glob(X_pca)['output'].squeeze(-1)
            else:
               pca = PCA(n_components=n)
               X_pca = pca.fit_transform(x)
               X_pca = torch.tensor(X_pca, dtype=torch.float32)
               output = self.model(X_pca)['output'].squeeze(-1)
            loss += self.loss(output, y)
            #test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            test_mse += torch.mean((output - y) ** 2).item()
            #@loss += self.loss(output, y)
            #print(self.id + ", Test Accuracy:", test_acc / y.shape[0] )
            #print(self.id + ", Test Loss:", loss)
        self.update_parameters(self.local_model)
        return test_mse, y.shape[0], loss


    def get_next_train_batch(self,maptrainloader,iter_maptrainloader):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(iter_maptrainloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            iter_maptrainloader = iter(maptrainloader)
            (X, y) = next(iter_maptrainloader)
        result = {'X': X, 'y': y}
        return result

    def get_next_test_batch(self):
        try:
            # Samples a new batch for personalizing
            (X, y) = next(self.iter_testloader)
        except StopIteration:
            # restart the generator if the previous generator is exhausted.
            self.iter_testloader = iter(self.testloader)
            (X, y) = next(self.iter_testloader)
        return (X, y)

    def save_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model, os.path.join(model_path, "user_" + self.id + ".pt"))

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        self.model = torch.load(os.path.join(model_path, "server" + ".pt"))
    
    @staticmethod
    def model_exists():
        return os.path.exists(os.path.join("models", "server" + ".pt"))
