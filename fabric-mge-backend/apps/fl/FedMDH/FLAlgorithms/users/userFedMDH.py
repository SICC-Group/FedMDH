import torch
import torch.nn.functional as F
import numpy as np
from apps.fl.FedMDH.FLAlgorithms.users.userbase import User
from apps.fl.FedMDH.utils.model_utils import create_model
import pandas as pd
import os
from apps.fl.FedMDH.FLAlgorithms.optimizers.fedoptimizer import FedProxOptimizer
from apps.fl.FedMDH.Het_Update import wass_loss

class RegressionTracker():
    def __init__(self, num_features):
        """
        初始化存储预测值的和以及平方和的变量。
        :param num_features: 回归任务中预测值的维度（通常是1，如果是多输出回归，则大于1）。
        """
        self.num_features = num_features
        self.pred_sum = torch.zeros(num_features)  # 用于存储预测值的和
        self.pred_square_sum = torch.zeros(num_features)  # 用于存储预测值平方的和
        self.count = 0  # 记录样本数量

    def update(self, predictions):
        """
        更新回归预测值的和以及平方和。
        :param predictions: shape = n_samples * num_features
        """
        # 确保 pred_sum 和 pred_square_sum 在累加时不会被原地修改
        self.pred_sum = self.pred_sum + predictions.sum(dim=0)  # 修改累加操作
        self.pred_square_sum = self.pred_square_sum + (predictions ** 2).sum(dim=0)  # 修改累加操作
        self.count += predictions.size(0)  # update smaples number

    def avg_and_var(self):

        if self.count == 0:
            raise ValueError("Count is zero, cannot compute mean and variance.")
        
        mean = self.pred_sum / self.count  
        variance = (self.pred_square_sum / self.count) - (mean ** 2)  
        
        return mean.detach(), variance.detach()

class UserFedMDH(User):
    def __init__(self,
                 args, id, model, generative_model,
                 train_data, test_data, latent_layer_idx,
                 use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)
        self.gen_batch_size = args.gen_batch_size
        self.generative_model = generative_model
        #self.latent_model=latent_model
        self.latent_layer_idx = latent_layer_idx
        self.num_features = 1
        self.RegressionTracker = RegressionTracker(self.num_features)



    def exp_lr_scheduler(self, epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
        """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
        lr= max(1e-4, init_lr * (decay ** (epoch // lr_decay_epoch)))
        return lr

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count

    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {label:1 for label in range(self.unique_labels)}

    def train(self, glob_iter,net_preproc,maptrainloader,iter_maptrainloader,g_glob,net,global_mean, global_variance, prior,user_dir,personalized=False, early_stop=100, regularization=True, verbose=False):
        #self.clean_up_counts()
        net.train()
        net_preproc.eval()
        self.generative_model.load_state_dict(g_glob)
        self.generative_model.eval()
        TEACHER_LOSS, LATENT_LOSS, PREDICTIVE_LOSS, REGDISTILL_LOSS = 0, 0, 0, 0
        teacher_losses, reg_losses, user_latent_losses = [], [], []
        RegressionTracker0 = RegressionTracker(self.num_features)
        RegressionTracker1= RegressionTracker(self.num_features)
        generative_lr=self.exp_lr_scheduler(glob_iter, decay=0.9, init_lr=0.0001)
        self.optimizer = torch.optim.Adam(net.parameters(), lr=generative_lr)

        for epoch in range(10):
            net.train()
            for i in range(self.K):
                self.optimizer.zero_grad()
                samples =self.get_next_train_batch(maptrainloader,iter_maptrainloader)
                X,y= samples['X'], samples['y']
                y=y.float()
                y = y.unsqueeze(-1)
                X1=net_preproc(X)
                model_result=net(X1.float())
                user_output_logp = model_result['output']
                RegressionTracker0.update(user_output_logp.detach())
                self.RegressionTracker.update(user_output_logp.detach())
                predictive_loss=self.loss(user_output_logp, y)
                mean, variance = RegressionTracker0.avg_and_var()
                loss_w=wass_loss(net_preproc, X, y, prior,optimize_projector=True)
                #### sample y and generate z
                if regularization and epoch < early_stop:
                    gen_output=self.generative_model(y, latent_layer_idx=self.latent_layer_idx,prior=prior)['output']
                    logit_given_gen=net(gen_output,start_layer_idx=0)['output']
                    RegressionTracker1.update(logit_given_gen.detach())
                    mean1, variance1 = RegressionTracker1.avg_and_var()
                    user_latent_loss= torch.mean(self.generative_model.dist_loss(logit_given_gen, y))
                    sampled_y=torch.randn(self.batch_size) * global_variance.sqrt() + global_mean
                    reg_loss = torch.mean(self.generative_model.dist_loss(user_output_logp, sampled_y.unsqueeze(-1)))

                    gen_result=self.generative_model(sampled_y, latent_layer_idx=self.latent_layer_idx,prior=prior)
                    gen_output=gen_result['output'] 
                    user_output_logp =net(gen_output,start_layer_idx=0)['output']
                    teacher_loss = torch.mean(self.generative_model.dist_loss(user_output_logp, sampled_y.unsqueeze(-1)))

                    loss=predictive_loss+ self.generative_alpha*teacher_loss+self.generative_beta*reg_loss+self.generative_beta*loss_w
                    TEACHER_LOSS+=teacher_loss
                    LATENT_LOSS+=user_latent_loss
                    PREDICTIVE_LOSS+=predictive_loss
                    REGDISTILL_LOSS+=reg_loss
                    teacher_losses.append(teacher_loss.item())
                    reg_losses.append(reg_loss)
                    user_latent_losses.append(user_latent_loss)
                else:
                    #### get loss and perform optimization
                    loss=predictive_loss
                loss.backward()
                update_model = True

                if update_model:
                   self.optimizer.step()#self.local_model)
        self.clone_model_paramenter(net.parameters(), self.local_model)
        if personalized:
            self.clone_model_paramenter(net.parameters(), self.personalized_model_bar)
        # self.lr_scheduler.step()
        if regularization and verbose:
            TEACHER_LOSS=TEACHER_LOSS.detach().numpy() / (self.local_epochs * self.K)
            LATENT_LOSS=LATENT_LOSS.detach().numpy() / (self.local_epochs * self.K)
            PREDICTIVE_LOSS=PREDICTIVE_LOSS.detach().numpy() / (self.local_epochs * self.K)
            REGDISTILL_LOSS=REGDISTILL_LOSS.detach().numpy() / (self.local_epochs * self.K)
            info='\nUser Teacher Loss={:.4f}'.format(TEACHER_LOSS)
            info+=', Latent Loss={:.4f}'.format(LATENT_LOSS)
            info+=', predictive Loss={:.4f}'.format(PREDICTIVE_LOSS)
            info+=', reg distill Loss={:.4f}'.format(REGDISTILL_LOSS)
            # print(info)
        reg_loss_values = [loss.item() for loss in reg_losses]
        user_latent_losses_values = [loss.item() for loss in user_latent_losses]
        losses = {
            "teacher_loss": teacher_losses,
            "reg_loss": reg_loss_values,
            "user_latent_loss": user_latent_losses_values
        }

        result_dir = os.path.join(user_dir, 'output_csv')
        os.makedirs(result_dir, exist_ok=True)
        csv_filename = os.path.join(result_dir, 'module3_losses.csv')

        df = pd.DataFrame(losses)

        if not os.path.exists(csv_filename):

            df.to_csv(csv_filename, mode='w', index=False, header=True)
        else:

            df.to_csv(csv_filename, mode='a', index=False, header=False)

        return net, teacher_losses, reg_losses, user_latent_losses
    

    def adjust_weights(self, samples):
        labels, counts = samples['labels'], samples['counts']
        #weight=self.label_weights[y][:, user_idx].reshape(-1, 1)
        np_y = samples['y'].detach().numpy()
        n_labels = samples['y'].shape[0]
        weights = np.array([n_labels / count for count in counts]) # smaller count --> larger weight
        weights = len(self.available_labels) * weights / np.sum(weights) # normalized
        label_weights = np.ones(self.unique_labels)
        label_weights[labels] = weights
        sample_weights = label_weights[np_y]
        return sample_weights


