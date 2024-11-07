from apps.fl.FedMDH.FLAlgorithms.users.userFedMDH import UserFedMDH
# from FLAlgorithms.users.userpFedGen import RegressionTracker
from apps.fl.FedMDH.FLAlgorithms.servers.serverbase import Server
from apps.fl.FedMDH.FLAlgorithms.trainmodel.RKLDivLoss import RKLDivLoss
from apps.fl.FedMDH.utils.model_utils import create_model, convert_data
from apps.fl.FedMDH.utils.model_utils import read_data, read_user_data, aggregate_user_data, create_generative_model, \
    create_discriminator_model
from apps.fl.FedMDH.Het_Update import Het_LocalUpdate, het_test_img_local_all, train_preproc, aggregate_models
from apps.fl.FedMDH.Het_Nets import get_preproc_model
from apps.fl.FedMDH.FLAlgorithms.users.userbase import User
from torch.utils.data import TensorDataset, DataLoader
from apps.fl.FedMDH.prior_reg import Prior
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from torchvision.utils import save_image
import os
import copy
import time
import itertools
import matplotlib.pyplot as plt
import csv
from apps.fl.FL_for_matdata.utils import dict_to_namespace, prepare_data, record, getcsv_mongo

MIN_SAMPLES_PER_LABEL = 1


class FedMDH(Server):
    def __init__(self, args, model, seed):
        super().__init__(args, model, seed)

        # Initialize data for all users
        self.data = read_data(args.dataset, args)
        # data contains: clients, groups, train_data, test_data, proxy_data
        clients = self.data[0]
        total_users = len(clients)
        self.total_test_samples = 0
        self.local = 'local' in self.algorithm.lower()  # false
        self.use_adam = 'adam' in self.algorithm.lower()  # false
        self.global_mean = None
        self.global_variance = None
        self.early_stop = 20  # stop using generated samples after 20 local epochs
        self.latent_model = create_model(32)
        # self.student_model = copy.deepcopy(self.latent_model) #cnn模型
        # self.student_model = copy.deepcopy(self.model) #cnn模型
        self.generator_model_glob = create_generative_model(args.dataset, args.algorithm, args.embedding)  # 生成器模型
        self.discriminator_model_glob = create_discriminator_model(args.dataset, args.algorithm,
                                                                   args.embedding)  # 判别器模型
        if not args.train:
            print('number of generator parameteres: [{}]'.format(self.generator_model_glob.get_number_of_parameters()))
            print('number of discriminator parameteres: [{}]'.format(
                self.discriminator_model_glob.get_number_of_parameters()))
            print('number of model parameteres: [{}]'.format(self.model.get_number_of_parameters()))
        self.latent_layer_idx = self.generator_model_glob.latent_layer_idx
        self.init_ensemble_configs()
        print("latent_layer_idx: {}".format(self.latent_layer_idx))
        # print("label embedding {}".format(self.generative_model.embedding))
        print("ensemeble learning rate: {}".format(self.ensemble_lr))
        print("ensemeble alpha = {}, beta = {}, eta = {}".format(self.ensemble_alpha, self.ensemble_beta,
                                                                 self.ensemble_eta))
        print("generator alpha = {}, beta = {}".format(self.generative_alpha, self.generative_beta))
        self.init_loss_fn()  # 初始化损失函数
        self.train_data_loader, self.train_iter = aggregate_user_data(self.data, args.dataset,
                                                                      self.ensemble_batch_size)  # 聚合用户数据(但不应该对用户数据进行聚合呀？)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generator_model_glob.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.discriminator_optimizer = torch.optim.Adam(
            params=self.discriminator_model_glob.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=self.weight_decay, amsgrad=False)
        self.generator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=0.98)
        self.discriminator_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.discriminator_optimizer, gamma=0.98)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.ensemble_lr, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.002)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.98)

        # 指定目录下创建 'results' 文件夹,用于存放结果图片
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 获取上一级目录
        parent_dir = os.path.dirname(current_dir)
        # 获取上上级目录
        grandparent_dir = os.path.dirname(parent_dir)
        self.result_dir = os.path.join(grandparent_dir, 'results/{taskID}'.format(taskID=args.taskID))
        # 确保 'result' 文件夹存在
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        #### creating users ####
        self.users = []
        for i in range(total_users):
            id, train_data, test_data = read_user_data(i, self.data, dataset=args.dataset, )
            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)
            # id, train, test=read_user_data(i, data, dataset=args.dataset)
            user = UserFedMDH(
                args, id, model, self.generator_model_glob,
                train_data, test_data, self.latent_layer_idx,
                use_adam=self.use_adam)
            self.users.append(user)
        print("Number of Train/Test samples:", self.total_train_samples, self.total_test_samples)
        print("Data from {} users in total.".format(total_users))
        print("Finished creating FedAvg server.")

    def select_top_features(self, x_train: torch.Tensor, y_train: torch.Tensor, num_features: int) -> torch.Tensor:
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

    def train(self, args):
        #### pretraining
        w_glob_keys = []
        mean_xtrains = {}
        # 初始化数据结构来存储每个用户的损失
        w_glob_keys = list(itertools.chain.from_iterable(w_glob_keys))
        net_glob = self.model.to(args.device)
        net_glob.train()  # 11111

        mean_train = torch.zeros(args.dim_latent)
        var_train = torch.zeros(args.dim_latent)
        net_preprocs = {}
        generator_locals = {}
        discriminator_locals = {}
        net_locals = {}
        user_data = {}
        map_data = {}
        map_xdata = {}
        user_Xtrain = {}
        user_ytrain = {}
        # 数据集预处理，各worker将训练集统一成同一个维度
        for user_idx, user in enumerate(self.users):
            net_preproc = get_preproc_model(args, dim_in=args.dim_in, dim_out=args.dim_latent)
            net_preprocs[user_idx] = (net_preproc)
            generator_model_local = create_generative_model(args.dataset, args.algorithm, args.embedding)  # 生成器模型
            generator_locals[user_idx] = (generator_model_local)
            discriminator_model_local = create_discriminator_model(args.dataset, args.algorithm,
                                                                   args.embedding)  # 判别器模型
            discriminator_locals[user_idx] = (discriminator_model_local)
            id = self.data[0][user_idx]
            train_data = self.data[2][id]
            X_train, y_train = convert_data(train_data['x'], train_data['y'], dataset=self.dataset)

            user_ytrain[user_idx] = y_train
            # dataset = TensorDataset(X_pca, y_train)
            top_features = self.select_top_features(X_train, y_train, args.dim_in)
            print("train_top_features", top_features)
            # 提取最相关的特征，赋值给 x_train1
            X_train1 = X_train[:, top_features]
            user_Xtrain[user_idx] = X_train1
            dataset = TensorDataset(X_train1, y_train)
            user_data[user_idx] = DataLoader(dataset=dataset, batch_size=args.preproc_batch_size, drop_last=True,
                                             shuffle=True)
            mean_train += torch.mean(X_train1, dim=0)  # 每一维的均值
            mean_xtrains[user_idx] = torch.mean(X_train1, dim=0)
            # var_train+= torch.var(X_pca, dim=0)
        # woker将本地的特征分布发送给server进行聚合，生成全局的特征分布，用于初始化全局锚点
        total_mean_train = mean_train / args.num_users
        mean_xtrains_tensor = torch.stack(list(mean_xtrains.values()))
        var_train = torch.var(mean_xtrains_tensor, dim=0, unbiased=False)
        net_glopreproc = get_preproc_model(args, dim_in=args.dim_in, dim_out=args.dim_latent)
        generator_glob = create_generative_model(args.dataset, args.algorithm, args.embedding)  # 生成器模型
        discriminator_glob = create_discriminator_model(args.dataset, args.algorithm, args.embedding)  # 判别器模型
        # 初始化全局锚点，分发给worker
        prior = Prior([args.dim_latent], total_mean_train)
        # 本地worker的映射模型预训练
        train_preproc(net_preprocs, user_data, prior,
                      n_epochs=args.align_epochs,
                      args=args,
                      verbose=True)
        # 每个worker构建映射后的样本，传递给CGAN模型，用于CGAN模型的预训练
        for user_idx, user in enumerate(self.users):
            map_xdata[user_idx] = net_glopreproc(user_Xtrain[user_idx])
            dataset = TensorDataset(map_xdata[user_idx], user_ytrain[user_idx])
            map_data[user_idx] = DataLoader(dataset=dataset, batch_size=args.preproc_batch_size, drop_last=True,
                                            shuffle=True)
        # 每个worker预训练本地的CGAN模型
        self.glob_train_generator(args, map_data, generator_locals, discriminator_locals, args.num_users, prior,
                                  args.n_epochs, latent_layer_idx=0)
        # server聚合所有worker的CGAN模型，生成全局的CGAN模型
        self.generator_model_glob = aggregate_models(generator_locals, generator_glob, args)
        self.discriminator_model_glob = aggregate_models(discriminator_locals, discriminator_glob, args)
        # 开始正式的联邦学习
        glob_losses = []
        for glob_iter in range(self.num_glob_iters):
            print("\n\n-------------Round number: ", glob_iter, " -------------\n\n")

            # im_out_list = []
            self.selected_users = self.select_users(glob_iter, self.num_users, return_idx=True)

            # 记录所有用户的本地模型参数之和

            mapdataset = {}
            maptrainloader = {}
            iter_maptrainloader = {}
            im_outs = {}
            useralldatay = {}
            # 记录选择的用户数量
            num_selected_users = len(self.selected_users)

            if not self.local:
                self.send_parameters(mode=self.mode)  # broadcast averaged prediction model
            # 这里是测试集计算MSE和R2,用于评估模型精度的代码
            glob_loss,glob_mae,glob_rmse,glob_r2,output,label = self.evaluate(net_glob, net_glopreproc,args=args)
            glob_losses.append(glob_loss * 1.0)
            # 这里是每个worker将本地标签分布发送给server
            self.send_logits()
            # chosen_verbose_user = np.random.randint(0, len(self.users))
            self.timestamp = time.time()  # log user-training start time
            # 这个for训练下的内容都是worker本地训练的内容
            for userid, user in enumerate(self.selected_users):  # allow selected users to train

                user_dir = os.path.join(self.result_dir, f'worker_{user.id}')
                os.makedirs(user_dir, exist_ok=True)

                local = Het_LocalUpdate(args=args, dataset=user_data[userid], current_iter=glob_iter)
                local.update_prior = args.update_prior

                last = glob_iter == args.num_glob_iters
                # 首先每个worker训练本地的映射模型
                im_out, y_out, w_prelocal = local.train(glob_iter,
                                                        # local.train参数
                                                        net=net_glob.to(args.device),
                                                        net_preproc=net_preprocs[userid].to(args.device),
                                                        w_glob_keys=w_glob_keys,
                                                        # CGAN参数
                                                        g_glob=self.generator_model_glob.state_dict(),
                                                        d_glob=self.discriminator_model_glob.state_dict(),
                                                        #######
                                                        # 中间参数
                                                        user=user,
                                                        userid=userid,
                                                        user_dir=user_dir,
                                                        #######
                                                        # user.train参数
                                                        global_mean=self.global_mean,
                                                        global_variance=self.global_variance,
                                                        personalized=self.personalized,
                                                        regularization=glob_iter > 0,
                                                        #######
                                                        prior=prior,
                                                        lr=args.lr, last=last
                                                        )
                im_outs[userid] = im_out
                net_preprocs[userid] = w_prelocal
                useralldatay[userid] = y_out
            # 本地锚点在server端聚合
            prior.mu = prior.mu_temp / prior.n_update
            prior.init_mu_temp()
            # 每个worker训练本地的CGAN模型
            for userid, user in enumerate(self.selected_users):
                d_loss, g_loss, g_local, d_local = self.local_train_generator(args, generator_locals[userid],
                                                                              discriminator_locals[userid], prior,
                                                                              glob_iter, args.num_users, user_dir,
                                                                              im_outs[userid], useralldatay[userid],
                                                                              latent_layer_idx=self.latent_layer_idx)
                mapdataset[userid] = TensorDataset(user_Xtrain[userid].detach(), useralldatay[userid])
                # mapdataset_nums = len(mapdataset)
                maptrainloader[userid] = DataLoader(mapdataset[userid], args.batch_size, shuffle=True, drop_last=True)
                iter_maptrainloader[userid] = iter(maptrainloader[userid])
                generator_locals[userid] = g_local
                discriminator_locals[userid] = d_local
            # 聚合CGAN模型
            self.generator_model_glob = aggregate_models(generator_locals, generator_glob, args)
            self.discriminator_model_glob = aggregate_models(discriminator_locals, discriminator_glob, args)
            # 每个worker训练本地的预测模型
            for userid, user in enumerate(self.selected_users):
                w_local, teacher_losses, reg_losses, latent_losses = user.train(glob_iter, net_preprocs[userid],
                                                                                maptrainloader[userid],
                                                                                iter_maptrainloader[userid],
                                                                                self.generator_model_glob.state_dict(),
                                                                                net_glob.to(args.device),
                                                                                self.global_mean, self.global_variance,
                                                                                prior, user_dir,
                                                                                personalized=self.personalized,
                                                                                early_stop=20,
                                                                                verbose=True and glob_iter > 0,
                                                                                regularization=glob_iter > 0)

                net_locals[userid] = w_local
            # 聚合预测模型
            self.aggregate_logits()
            net_glopreproc = aggregate_models(net_preprocs, net_glopreproc, args)
            net_glob = aggregate_models(net_locals, net_glob, args)
            
        self.save_results(args)
        self.save_model()

        # glob_loss, glob_mae,glob_rmse,glob_r2分别是MSE, MAE, RMSE, R2的接口数据
        print("MSE = {:.4f}, MAE = {:.4f}, RMSE={:.4f},r2={:4f}.".format(glob_loss, glob_mae,glob_rmse,glob_r2))
        # record loss
        all_loss = {
            "MSE": glob_loss,
            "MAE": glob_mae,
            "RMSE": glob_rmse,
            "R2": glob_rmse
        }
        with open(os.path.join(self.result_dir, "last_loss.txt"), 'w') as f:
            f.write("Test metrics of final global model:\n{}".format(all_loss))
        # Plot test_data Loss
        plt.figure(figsize=(7, 5))
        plt.plot(glob_losses, label='Test Data Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Test Data Loss Curves')
        plt.legend()
        plot_filename = os.path.join(self.result_dir, 'test_data_loss.png')
        plt.savefig(plot_filename)
        print(f"Loss curves saved at: {plot_filename}")
        plt.show()
        plt.figure(figsize=(7, 5))

        # 绘制各组散点图并在图例中显示R²值
        plt.scatter(label.detach().numpy(), output.detach().numpy(), c='blue', s=50, alpha=0.6, label=f'FedMDH')

        # 设置坐标轴标签和图例
        plt.xlabel('Test Labels (eV/atom)', fontsize=12)
        plt.ylabel('Predicted Values (eV/atom)', fontsize=12)
        plt.tick_params(axis='x', labelsize=12)  # 设置 x 轴刻度标签的字体大小
        plt.tick_params(axis='y', labelsize=12)
        plt.legend(fontsize=12)
        plot_filename = os.path.join(self.result_dir, 'predicted_vs_actual.png')
        plt.savefig(plot_filename)
        print(f"Loss curves saved at: {plot_filename}")
        plt.show()

    def set_requires_grad(self, model, requires_grad=True):
        for param in model.parameters():
            param.requires_grad = requires_grad

    def local_train_generator(self, args, generator_local, discriminator_local, prior, glob_iter, num_users, user_dir,
                              im_out, target, latent_layer_idx=0):
        """
        本地训练生成器和判别器，返回g_loss与d_loss
        """

        self.generator_model_glob.train()
        self.discriminator_model_glob.train()

        g_losses = []
        d_losses = []
        idxs_users = np.arange(num_users)

        # data_adapt = user_data[idx]
        generator_projector = copy.deepcopy(generator_local.to(args.device))
        self.set_requires_grad(generator_projector, requires_grad=True)
        discriminator_projector = copy.deepcopy(discriminator_local.to(args.device))
        self.set_requires_grad(discriminator_projector, requires_grad=True)
        generator_optimizer = torch.optim.Adam(params=generator_projector.parameters(),
                                               lr=self.ensemble_lr, betas=(0.9, 0.999), eps=1e-08,
                                               weight_decay=self.weight_decay, amsgrad=False)
        discriminator_optimizer = torch.optim.Adam(params=discriminator_projector.parameters(),
                                                   lr=self.ensemble_lr, betas=(0.9, 0.999), eps=1e-08,
                                                   weight_decay=self.weight_decay, amsgrad=False)

        for itm in range(10):
            generator_optimizer.zero_grad()

            y = target

            ## feed to generator
            gen_result = generator_projector(y, latent_layer_idx=latent_layer_idx, prior=prior, verbose=True)
            # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
            gen_output, eps = gen_result['output'], gen_result['eps']
            # print("gen_output:",gen_output.shape)

            # Pass generated data through discriminator
            fake_score = discriminator_projector(gen_output, y)

            # Again pass generated data through discriminator
            g_loss = torch.mean(torch.nn.functional.mse_loss(fake_score, torch.ones_like(fake_score)))
            g_loss.backward()
            generator_optimizer.step()
            g_losses.append(g_loss.item())

            discriminator_optimizer.zero_grad()

            # 进行判别器的训练
            real_score = discriminator_projector(im_out.detach(), y)
            fake_score = discriminator_projector(gen_output.detach(), y)
            d_loss_real = torch.mean(torch.nn.functional.mse_loss(real_score, torch.ones_like(real_score)))
            d_loss_fake = torch.mean(torch.nn.functional.mse_loss(fake_score, torch.zeros_like(fake_score)))

            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_loss.backward()
            discriminator_optimizer.step()
            d_losses.append(d_loss.item())
            # print(f"User {ind}, Epoch {itm}, Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}")
        self.set_requires_grad(generator_projector, requires_grad=False)
        generator_local = copy.deepcopy(generator_projector)
        self.set_requires_grad(discriminator_projector, requires_grad=False)
        discriminator_local = copy.deepcopy(discriminator_projector)

        return d_losses, g_losses, generator_local, discriminator_local

    def glob_train_generator(self, args, user_data, generator_locals, discriminator_locals, num_users, prior, n_epochs,
                             latent_layer_idx=0):
        """
        服务器预训练生成器和判别器
        """

        def update_glob_generator_():

            self.generator_model_glob.train()
            self.discriminator_model_glob.train()

            g_losses = []
            d_losses = []
            idxs_users = np.arange(num_users)

            for ind, idx in enumerate(idxs_users):
                data_adapt = user_data[idx]
                generator_projector = copy.deepcopy(generator_locals[idx].to(args.device))
                self.set_requires_grad(generator_projector, requires_grad=True)
                discriminator_projector = copy.deepcopy(discriminator_locals[idx].to(args.device))
                self.set_requires_grad(discriminator_projector, requires_grad=True)
                generator_optimizer = torch.optim.Adam(params=generator_projector.parameters(),
                                                       lr=self.ensemble_lr, betas=(0.9, 0.999), eps=1e-08,
                                                       weight_decay=self.weight_decay, amsgrad=False)
                discriminator_optimizer = torch.optim.Adam(params=discriminator_projector.parameters(),
                                                           lr=self.ensemble_lr, betas=(0.9, 0.999), eps=1e-08,
                                                           weight_decay=self.weight_decay, amsgrad=False)

                for itm in range(n_epochs):
                    for data, target in data_adapt:
                        generator_optimizer.zero_grad()

                        y = target

                        ## feed to generator
                        gen_result = generator_projector(y, latent_layer_idx=latent_layer_idx, prior=prior,
                                                         verbose=True)
                        # get approximation of Z( latent) if latent set to True, X( raw image) otherwise
                        gen_output, eps = gen_result['output'], gen_result['eps']
                        # print("gen_output:",gen_output.shape)

                        # Pass generated data through discriminator
                        fake_score = discriminator_projector(gen_output, y)

                        # Again pass generated data through discriminator
                        g_loss = torch.mean(torch.nn.functional.mse_loss(fake_score, torch.ones_like(fake_score)))
                        g_loss.backward()
                        generator_optimizer.step()
                        g_losses.append(g_loss.item())

                        discriminator_optimizer.zero_grad()

                        # 进行判别器的训练
                        real_score = discriminator_projector(data.detach(), y)
                        fake_score = discriminator_projector(gen_output.detach(), y)
                        d_loss_real = torch.mean(torch.nn.functional.mse_loss(real_score, torch.ones_like(real_score)))
                        d_loss_fake = torch.mean(torch.nn.functional.mse_loss(fake_score, torch.zeros_like(fake_score)))

                        d_loss = 0.5 * (d_loss_real + d_loss_fake)
                        d_loss.backward()
                        discriminator_optimizer.step()
                        d_losses.append(d_loss.item())
                        # print(f"User {ind}, Epoch {itm}, Generator Loss: {g_loss:.4f}, Discriminator Loss: {d_loss:.4f}")
                self.set_requires_grad(generator_projector, requires_grad=False)
                generator_locals[idx] = copy.deepcopy(generator_projector)
                self.set_requires_grad(discriminator_projector, requires_grad=False)
                discriminator_locals[idx] = copy.deepcopy(discriminator_projector)
            return d_losses, g_losses

        #
        d_losses, g_losses = update_glob_generator_()

        # 保存 g_loss 和 d_loss 为 CSV 文件
        csv_filename = os.path.join(self.result_dir, 'glob_generator_pretrain_losses.csv')
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Iteration', 'Generator Loss', 'Discriminator Loss'])
            for i, (g_loss, d_loss) in enumerate(zip(g_losses, d_losses)):
                writer.writerow([i, g_loss, d_loss])

        # Plotting g_loss and d_loss curves
        # plt.figure(figsize=(12, 6))
        #
        # # Plot Generator Loss
        # plt.subplot(2, 1, 1)
        # plt.plot(g_losses, label='Generator Loss')
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.title('Generator Loss over Iterations')
        # plt.legend()
        #
        # # Plot Discriminator Loss
        # plt.subplot(2, 1, 2)
        # plt.plot(d_losses, label='Discriminator Loss')
        # plt.xlabel('Iteration')
        # plt.ylabel('Loss')
        # plt.title('Discriminator Loss over Iterations')
        # plt.legend()
        #
        # # Save the plots
        # plot_filename = os.path.join(self.result_dir, 'glob_generator_pretrain_loss_curves.png')
        # plt.savefig(plot_filename)
        # plt.close()
        #
        # print(f"Loss curves saved at: {plot_filename}")
        #
        # info = "Generator Loss= {:.4f}, Discriminator Loss= {:.4f}, ". \
        #     format(g_loss, d_loss)
        # print(info)

    def aggregate_logits(self, selected=True):
        sum_means = 0
        sum_variances = 0
        users = self.selected_users if selected else self.users

        for user in users:
            mean, variance = user.RegressionTracker.avg_and_var()  # 获取均值和方差
            sum_means += mean
            sum_variances += variance

        # 计算全局的均值和方差
        self.global_mean = sum_means / len(users)
        self.global_variance = sum_variances / len(users)

    def send_logits(self):
        if self.global_mean is None or self.global_variance is None: return
        for user in self.selected_users:
            user.global_mean = self.global_mean.clone().detach()
            user.global_variance = self.global_variance.clone().detach()

    def get_label_weights(self):
        label_weights = []
        qualified_labels = []
        for label in range(self.unique_labels):
            weights = []
            for user in self.selected_users:
                weights.append(user.label_counts[label])
            if np.max(weights) > MIN_SAMPLES_PER_LABEL:
                qualified_labels.append(label)
            # uniform
            label_weights.append(np.array(weights) / np.sum(weights))
        label_weights = np.array(label_weights).reshape((self.unique_labels, -1))
        return label_weights, qualified_labels

    def visualize_images(self, generator, glob_iter, repeats=1):
        """
        Generate and visualize data for a generator.
        """
        os.system("mkdir -p images")
        path = f'images/{self.algorithm}-{self.dataset}-iter{glob_iter}.png'
        y = self.available_labels
        y = np.repeat(y, repeats=repeats, axis=0)
        y_input = torch.tensor(y)
        generator.eval()
        images = generator(y_input, latent=False)['output']  # 0,1,..,K, 0,1,...,K
        images = images.view(repeats, -1, *images.shape[1:])
        images = images.view(-1, *images.shape[2:])
        # save_image(images.detach(), path, nrow=repeats, normalize=True)
        print("Image saved to {}".format(path))
