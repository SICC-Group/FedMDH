import math
import torch.linalg as linalg
from apps.fl.FedMDH.utils.model_utils import create_model,convert_data
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import copy
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from apps.fl.FedMDH.utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model, create_discriminator_model
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad

def dist_torch(x1,x2):
    x1p = x1.pow(2).sum(1).unsqueeze(1)
    x2p = x2.pow(2).sum(1).unsqueeze(1)
    prod_x1x2 = torch.mm(x1,x2.t())
    distance = x1p.expand_as(prod_x1x2) + x2p.t().expand_as(prod_x1x2) -2*prod_x1x2
    return distance #/x1.size(0)/x2.size(0) 


def calculate_mmd(X,Y):
    def my_cdist(x1, x2):
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
        res = torch.addmm(x2_norm.transpose(-2, -1),
                          x1,
                          x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
        return res.clamp_min_(1e-30)
    
    def gaussian_kernel(x, y, gamma=[0.0001,0.001, 0.01, 0.1, 1, 10, 100]):
        D = my_cdist(x, y)
        K = torch.zeros_like(D)
    
        for g in gamma:
            K.add_(torch.exp(D.mul(-g)))
    
        return K
    

    Kxx = gaussian_kernel(X, X).mean()
    Kyy = gaussian_kernel(Y, Y).mean()
    Kxy = gaussian_kernel(X, Y).mean()
    return Kxx + Kyy - 2 * Kxy


def calculate_2_wasserstein_dist(X, Y):


    if X.shape != Y.shape:
        raise ValueError("Expecting equal shapes for X and Y!")

    # the linear algebra ops will need some extra precision -> convert to double
    X, Y = X.transpose(0, 1).double(), Y.transpose(0, 1).double()  # [n, b]
    mu_X, mu_Y = torch.mean(X, dim=1, keepdim=True), torch.mean(Y, dim=1, keepdim=True)  # [n, 1]
    n, b = X.shape
    fact = 1.0 if b < 2 else 1.0 / (b - 1)

    # Cov. Matrix
    E_X = X - mu_X
    E_Y = Y - mu_Y
    cov_X = torch.matmul(E_X, E_X.t()) * fact  # [n, n]
    cov_Y = torch.matmul(E_Y, E_Y.t()) * fact

    # calculate Tr((cov_X * cov_Y)^(1/2)). with the method proposed in https://arxiv.org/pdf/2009.14075.pdf
    # The eigenvalues for M are real-valued.
    C_X = E_X * math.sqrt(fact)  # [n, n], "root" of covariance
    C_Y = E_Y * math.sqrt(fact)
    M_l = torch.matmul(C_X.t(), C_Y)
    M_r = torch.matmul(C_Y.t(), C_X)
    M = torch.matmul(M_l, M_r)
    # print(M)
    S = linalg.eigvals(M+1e-6) + 1e-15  # add small constant to avoid infinite gradients from sqrt(0)
    sq_tr_cov = S.sqrt().abs().sum()

    # plug the sqrt_trace_component into Tr(cov_X + cov_Y - 2(cov_X * cov_Y)^(1/2))
    trace_term = torch.trace(cov_X + cov_Y) - 2.0 * sq_tr_cov  # scalar

    # |mu_X - mu_Y|^2
    diff = mu_X - mu_Y  # [n, 1]
    mean_term = torch.sum(torch.mul(diff, diff))  # scalar

    # put it together
    return (trace_term + mean_term).float()


def wass_loss(net_projector, data, target, prior, optimize_projector=True, distance='wd'):

    loss = torch.tensor(0.0, requires_grad = True)
     
    out = net_projector(data)
    mean_t = prior.mu  # Assuming we use the first component as a reference
    var_t = prior.logvar
            
    prior_samples = prior.sampling_gaussian(out.shape[0], mean_t, var_t)
    # print("#",prior_samples.shape)
    # print(prior_samples)
    if distance == 'wd':
        loss_dist = calculate_2_wasserstein_dist(out, prior_samples)
        loss = torch.add(loss, loss_dist, alpha = 1)
    elif distance == 'mmd':
        loss_dist = calculate_mmd(out, prior_samples)
        loss = torch.add(loss, loss_dist, alpha = 1)

    return loss

def aggregate_models(net_preprocs, global_model, args):
    # 初始化全局模型的权重
    global_state_dict = copy.deepcopy(net_preprocs[0].state_dict())

    # 初始化聚合参数为 0
    for key in global_state_dict.keys():
        global_state_dict[key] = torch.zeros_like(global_state_dict[key])

    # 遍历每个用户的模型，逐步求和
    for i in range(args.num_users):
        local_state_dict = net_preprocs[i].state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] += local_state_dict[key]  # 逐用户求和
    
    # 按用户数量求平均
    for key in global_state_dict.keys():
        global_state_dict[key] = global_state_dict[key].float() / args.num_users
    
    # 将聚合后的参数赋值给全局模型
    global_model.load_state_dict(global_state_dict)

    return global_model


def train_preproc(net_preprocs, user_data, prior,n_epochs,args=None,verbose=True):
    
    prior.mu = prior.mu.to(args.device)  
    prior.logvar = prior.logvar.to(args.device)  
    if args.device == 'cuda':
        pin_memory = True
    else:
        pin_memory = False

    # for each user, optimize the loss
    idxs_users = np.arange(args.num_users)

    for ind, idx in enumerate(idxs_users):
        # data_adapt = DataLoader(dataset=dataset_train, batch_size=args.align_bs, shuffle=True
        #                             ,drop_last=False,pin_memory=pin_memory,
        #                             num_workers=args.num_workers)
        data_adapt = user_data[idx]
        # data_iter = iter(dataset_train)
        net_projector =  copy.deepcopy(net_preprocs[idx].to(args.device))
        set_requires_grad(net_projector, requires_grad=True)
        
        optimizer_preproc = torch.optim.Adam(net_projector.parameters(),lr=args.align_lr)

        for itm in range(n_epochs):
            loss_tot = 0
            for data, target in data_adapt:
                data = data.to(args.device)  
                target = target.to(args.device) 
                target=target.view(-1, 1)   
                # data.requires_grad_(True)  
                # target.requires_grad_(True)            
                optimizer_preproc.zero_grad()
                net_projector.zero_grad()
                loss = wass_loss(net_projector, data, target, prior,optimize_projector=True,distance=args.distance)
                loss.backward()
                optimizer_preproc.step()

                loss_tot +=loss.item()
            if verbose :
                print(f"User {ind}, Epoch {itm}, Loss: {loss_tot:.4f}")
        set_requires_grad(net_projector, requires_grad=False)
        net_preprocs[idx] = copy.deepcopy(net_projector)



class Het_LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None,
                 mean_target=None,current_iter= 1000):
        self.args = args
        #移除交叉熵损失，换成均方误差损失
        #self.loss_func = nn.CrossEntropyLoss()   
        self.loss_func = nn.MSELoss() 
        self.ldr_train = DataLoader(dataset, batch_size=self.args.local_bs, shuffle=True,
                                    pin_memory=True,
                                    num_workers=args.num_workers)
        #self.ldr_train = dataset
        self.dataset = dataset
        self.idxs = idxs
        self.indd = indd
        self.update_prior = args.update_prior > 0
        self.update_net_preproc = args.update_net_preproc > 0
        self.update_global_representation = current_iter > args.start_optimize_rep

        self.ensemble_lr = 1e-3
        self.weight_decay = 1e-2
    

        # 生成器和判别器
        self.generator_model = create_generative_model(args.dataset, args.algorithm, args.embedding)#生成器模型
        self.discriminator_model = create_discriminator_model(args.dataset, args.algorithm, args.embedding) # 判别器模型
        self.latent_layer_idx = self.generator_model.latent_layer_idx
        self.generator_optimizer = torch.optim.Adam(
            params=self.generator_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.discriminator_optimizer = torch.optim.Adam(
            params=self.discriminator_model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generative_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generator_optimizer, gamma=0.98)
        self.discriminator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.discriminator_optimizer, gamma=0.98)
    
    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):

        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr
    
    def train(self, glob_iter,net, net_preproc, w_glob_keys, g_glob, d_glob, user, userid, user_dir, global_mean, global_variance, personalized, regularization, last=False, dataset_test=None, 
              prior=None,ind=-1, idx=-1, lr=0.1):
        bias_p=[]
        weight_p=[]
        previous_grads={name: torch.zeros_like(param) for name, param in net.named_parameters()}
        g_losses = []
        d_losses = []
        generator_locals={}
        discriminator_locals={}
        #net_projector =  copy.deepcopy(net.to(self.args.device))
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
 
        generative_lr=self.exp_lr_scheduler(glob_iter, decay=0.9, init_lr=0.0001)    
        optimizer = torch.optim.Adam(net.parameters(), lr=generative_lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
        local_eps = self.args.local_epochs
        if last:
            local_eps =  max(10,local_eps-self.args.local_rep_ep)
        if self.update_global_representation:
            head_eps = local_eps-self.args.local_rep_ep
        else:
            head_eps = local_eps
        epoch_loss = []
        num_updates = 0
        mu_local = nn.Parameter(prior.mu.clone(), requires_grad=True)
        optim_mean = torch.optim.Adam([mu_local],lr=0.0001)
        # scheduler_mean=torch.optim.lr_scheduler.ExponentialLR(optimizer=optim_mean, gamma=0.9)
        net.to(self.args.device)
        net_preproc.to(self.args.device)
        mu_local = mu_local.to(self.args.device)
        prior.mu_temp = prior.mu_temp.to(self.args.device)  
        optimizer_preproc = torch.optim.Adam(net_preproc.parameters(),lr=self.args.align_lr)
        scheduler_preproc=torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer_preproc, gamma=0.9)
        clip_value=3
        threshold=0.1
        total_norm=0
        # 10
        g_locals = {key: g_glob[key].clone() for key in g_glob}
        d_locals = {key: d_glob[key].clone() for key in d_glob}
        for loiter in range(local_eps):
            net_preproc.train()
            net.eval()
            #net_projector.eval()
            batch_loss = []
            
            for batch_idx, (data, target) in enumerate(self.dataset):
                # first 9 round
                if (loiter < head_eps ) or last or not w_glob_keys:
                    # update net_preproc parameter
                    if self.update_net_preproc:
                        set_requires_grad(net_preproc, requires_grad=True)
                    # not update net parameter
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                #last round
                elif (loiter >= head_eps ):
                    # not update net_preproc parameter
                    set_requires_grad(net_preproc, requires_grad=False)
                    # update net parameter
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                im_out = net_preproc(data)
                #log_probs = net(im_out)
                predictions = net(im_out.float())['output']
                #loss_regression = self.loss_func(log_probs)
                #target=target.view(-1, 1)y = y.unsqueeze(-1)
                target = target.unsqueeze(-1)
                target=target.float()
                loss_regression = self.loss_func(predictions, target)
                optimizer_preproc.zero_grad()
                if self.update_net_preproc:
                    loss_W = wass_loss(net_preproc, data, target, prior,optimize_projector=True)

                else:
                    loss_W = 0
                loss_ref_dist = 0

                prior_samples = prior.sampling_gaussian(data.shape[0], prior.mu, prior.logvar)
                predictions_prior_samples = net(prior_samples.float())['output']
                loss_ref_dist = self.loss_func(predictions_prior_samples, target.float())
                num_updates += 1
                loss =  self.args.reg_w*loss_regression + loss_W + self.args.reg_reg_prior*loss_ref_dist
                batch_loss.append(loss.item())
                # optimizer.zero_grad()
                
                loss.backward()
               
                if self.update_net_preproc:
                    optimizer_preproc.step()
                if self.update_prior:
                    prior.mu_local = mu_local
                    lossW = wass_loss(net_preproc,data, target, prior,optimize_projector=False)
                    set_requires_grad(net, requires_grad=False)
                    #log_probs = net(mu_local)  
                    predictions_probs = net(mu_local)['output']
                    #predictions_probs = predictions_probs.detach()
                    target_mean=target.mean(dim=0)
                    #labels = torch.Tensor([ii for ii in range(self.args.num_classes)]).long()
                    #labels = labels.to(self.args.device)
                    loss = self.loss_func(predictions_probs, target_mean.float()) + lossW
                    optim_mean.zero_grad()
                    loss.backward()
                    optim_mean.step()
                    for name, param in net.named_parameters():
                        if name in w_glob_keys:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True   

            useralldata=self.dataset
            im_out = net_preproc(useralldata.dataset.tensors[0])
            
            
        set_requires_grad(net_preproc, requires_grad=True)
        #set_requires_grad(net, requires_grad=True)
        prior.mu_temp += mu_local.detach()
        prior.n_update += 1

        return im_out, useralldata.dataset.tensors[1],net_preproc

    
def plot_gen_output_distribution(gen_output, prior, user_id, user_dir):
    gen_output_np = gen_output.cpu().detach().numpy()

    plt.figure(figsize=(12, 6))
    for sample_idx in range(gen_output_np.shape[0]):
        feature_values = gen_output_np[sample_idx, :]
        plt.scatter(np.arange(1, gen_output_np.shape[1] + 1), feature_values, c='red', alpha=0.5, s=5)

    mu_np = prior.mu.cpu().detach().numpy()
    plt.scatter(np.arange(1, mu_np.shape[0] + 1), mu_np, c='green', marker='*', s=50, alpha=0.5)

    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.title(f'Generated Output Distribution with Global Anchor Points (User {user_id})')
    #plt.legend()
    
    os.makedirs(user_dir, exist_ok=True)
    plt.savefig(os.path.join(user_dir, 'gen_output_feature_distribution.png'))
    plt.close()


def plot_im_out_extended_distribution(im_out, im_out_extended, prior, user_id, user_dir):
    im_out_np = im_out.cpu().detach().numpy()
    im_out_extended_np = im_out_extended.cpu().detach().numpy()

    plt.figure(figsize=(12, 6))
    # 绘制 im_out 的分布
    for sample_idx in range(im_out_np.shape[0]):
        feature_values = im_out_np[sample_idx, :]
        plt.scatter(np.arange(1, im_out_np.shape[1] + 1), feature_values, c='red', alpha=0.5, s=5, label='im_out' if sample_idx == 0 else "", marker='o')
    # 绘制 im_out_extended 的分布
    for sample_idx in range(im_out_extended_np.shape[0]):
        feature_values = im_out_extended_np[sample_idx, :]
        plt.scatter(np.arange(1, im_out_extended_np.shape[1] + 1), feature_values, c='blue', alpha=0.5, s=5, label='im_out_extended' if sample_idx == 0 else "", marker='x')

    mu_np = prior.mu.cpu().detach().numpy()
    plt.scatter(np.arange(1, mu_np.shape[0] + 1), mu_np, c='green', marker='*', s=50, alpha=0.5)

    plt.xlabel('Feature Index')
    plt.ylabel('Feature Value')
    plt.title(f'Feature Distribution with Global Anchor Points (User {user_id})')
    plt.legend()
    
    os.makedirs(user_dir, exist_ok=True)
    plt.savefig(os.path.join(user_dir, 'im_out_extended_feature_distribution.png'))
    plt.close()

def plot_fake_scores(fake_scores,user_id, user_dir):
    fake_scores_np = [fs.cpu().detach().numpy() for fs in fake_scores]

    plt.figure(figsize=(12, 6))
    plt.plot(fake_scores_np, label='Fake Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title(f'Scores for User {user_id},related to g_loss')
    plt.legend()
    
    os.makedirs(user_dir, exist_ok=True)
    plt.savefig(os.path.join(user_dir, 'fake_scores_curve.png'))
    plt.close()

def plot_scores(fake_scores, real_scores, user_id, user_dir):
    fake_scores_np = [fs.cpu().detach().numpy() for fs in fake_scores]
    real_scores_np = [rs.cpu().detach().numpy() for rs in real_scores]

    plt.figure(figsize=(12, 6))
    plt.plot(fake_scores_np, label='Fake Score')
    plt.plot(real_scores_np, label='Real Score')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.title(f'Scores for User {user_id},related to d_loss')
    plt.legend()
    
    os.makedirs(user_dir, exist_ok=True)
    plt.savefig(os.path.join(user_dir, 'scores_curve.png'))
    plt.close()

def select_top_features(x_train: torch.Tensor, y_train: torch.Tensor, num_features: int) -> torch.Tensor:
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

def het_test_img_local(net_g, net_preproc, user_data, args,idx=None,indd=None, user_idx=-1, idxs=None):
    net_g.eval()
    net_preproc.eval()
    test_loss = 0
    #correct = 0
    net_preproc.to(args.device)
    net_g.to(args.device)
    

    count = 0
    #data_loader = DataLoader(user_data, batch_size=200, shuffle=True,drop_last=False)
    data_loader = user_data
    for idx, (data, target) in enumerate(data_loader):

        if args.gpu != -1:
            data, target = data.to(args.device), target.to(args.device)
            
        with torch.no_grad():

            top_features = select_top_features(data, target, args.dim_in)

                # 提取最相关的特征，赋值给 x_train1
            x1 = data[:, top_features]
            test_out=net_preproc(x1)
            prediction = net_g(test_out)
        target=target.unsqueeze(-1)
        #log_probs = net_g(net_preproc(data))
        # sum up batch loss
        #test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        batch_loss = F.mse_loss(prediction['output'], target.float(), reduction='sum').item()
        test_loss += batch_loss
        #y_pred = log_probs.data.max(1, keepdim=True)[1]
        #correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        count += data.shape[0]

        if idx==0:
            target_all = target.detach().cpu()
            y_pred_all = prediction['output'].detach().cpu()
        else:
            target_all = torch.cat((target_all,target.detach().cpu()),dim=0)
            y_pred_all = torch.cat((y_pred_all,prediction['output'].detach().cpu()),dim=0)
                
    test_loss /= count
    #accuracy = 100.00 * float(correct) / count
    #
    # bal_acc = 100*balanced_accuracy_score(target_all, y_pred_all.long())
    mse = mean_squared_error(target_all.numpy(), y_pred_all.numpy())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_all.numpy(), y_pred_all.numpy())
    r2 = r2_score(target_all.numpy(), y_pred_all.numpy())
    
    
    net_g.train()
    net_preproc.train()
    
    return  mse, rmse, test_loss, mae, r2 

def het_test_img_local_all(net, net_preprocs, args, users_test_data,dataset_train=None,dict_users_train=None, return_all=False):
    tot = 0
    num_idxxs = args.num_users
    #acc_test_local = np.zeros(num_idxxs)
    #loss_test_local = np.zeros(num_idxxs)
    #bal_acc_test_local = np.zeros(num_idxxs)
    mse_test_local = np.zeros(num_idxxs)
    rmse_test_local = np.zeros(num_idxxs)
    mae_test_local = np.zeros(num_idxxs)
    r2_test_local = np.zeros(num_idxxs)

    for idx in range(num_idxxs):

        net.eval()
        net_preproc = net_preprocs
        net_preproc.eval()

        mse, rmse, test_loss, mae, r2= het_test_img_local(net,net_preproc, users_test_data[idx], args, user_idx=idx) 
        
        dataset_test = users_test_data[idx].dataset
        test_tensors = dataset_test.tensors
        n_test = test_tensors[0].shape[0]
        # n_test = users_test_data[idx].tensors[0].shape[0]
        # n_test = len(users_test_data[idx].idxs)
        # n_test = users_test_data[idx].shape[0]
        tot += n_test
        mse_test_local[idx] = mse * n_test
        rmse_test_local[idx] = rmse * n_test
        mae_test_local[idx] = mae * n_test
        r2_test_local[idx] = r2 * n_test
        #bal_acc_test_local[idx] = c*n_test

        # del net_local
        net_preproc.train()
        net.train()
    
    avg_mse = sum(mse_test_local) / tot
    avg_rmse = sum(rmse_test_local) / tot
    avg_mae = sum(mae_test_local) / tot
    avg_r2 = sum(r2_test_local) / tot
    if return_all:
        return mse_test_local, rmse_test_local, test_loss, mae_test_local, r2_test_local
    
    return  avg_mse, avg_rmse, test_loss, avg_mae, avg_r2


