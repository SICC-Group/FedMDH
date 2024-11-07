#!/usr/bin/env python
import argparse
import json

import os

FEDMDH_PYTHON_PATH = "D:/Anaconda/Anaconda3/envs/fabric-mge-backend/python.exe"
FEDMDH_PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

from apps.fl.FL_for_matdata.utils import getcsv_mongo
from apps.fl.FedMDH.FLAlgorithms.servers.serverFedMDH import FedMDH
from apps.fl.FedMDH.utils.model_utils import create_model
import torch
from multiprocessing import Pool
import pandas as pd
import matplotlib.pyplot as plt



def plot_test_loss(args):
    # 初始化图形
    plt.figure(figsize=(12, 8))

    accepted_orgs = json.loads(args.accepted_orgs)

    # 遍历每个用户目录
    # for user_id in range(args.num_users):
    for user_name in accepted_orgs:
        # 定义 CSV 文件路径
        user_dir = os.path.join(args.result_path, f'user_{user_name}')
        csv_filename = os.path.join(user_dir, f'user_{user_name}_test_loss.csv')


        # 检查文件是否存在
        if os.path.exists(csv_filename):
            # 读取 CSV 文件
            df = pd.read_csv(csv_filename)
            # 生成轮次（序号）
            epochs = range(len(df))  # 轮次就是行数的序列           
            # 提取损失值
            losses = df.iloc[:, 0]  # 只有一列，所以直接取第一列

            # 绘制损失曲线
            plt.plot(epochs, losses, label=f'User {user_dir}')  # 根据文件夹名称生成标签
        else:
            print(f"File {csv_filename} does not exist.")

        # 配置图形
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        # plt.title(f'User {user_id} Test Loss')
        plt.title(f'User {user_name} Test Loss')
        # plt.legend()
        plt.grid(True)

        # 保存和显示图形
        plot_filename = os.path.join(user_dir, 'test_loss_comparison.png')
        plt.savefig(plot_filename)
        plt.close()
        # plt.show()

        # print(f"Loss comparison plot saved at: {plot_filename}")


def create_server_n_user(args, i):
    model = create_model(args.dim_latent)  # 改成MLP

# create pt
    # 数据集 fedmdh0read_data ->create_ptfile
    #结果serverbase.py Class SERVER TEST

    server = FedMDH(args, model, i)
    # else:
    # print("Algorithm {} has not been implemented.".format(args.algorithm))
    # exit()
    return server


def run_job(args, i):
    torch.manual_seed(i)
    print("\n\n         [ Start training iteration {} ]           \n\n".format(i))
    # Generate model
    server = create_server_n_user(args, i)
    if args.train:
        server.train(args)
        # server.test()
        # plot_test_loss(args)


def main(args):
    for i in range(args.times):
        run_job(args, i)
    print("Finished training.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Material")

    parser.add_argument("--train", type=int, default=1, choices=[0, 1])
    parser.add_argument("--algorithm", type=str, default="FedMDH")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--preproc_batch_size", type=int, default=128)
    parser.add_argument("--gen_batch_size", type=int, default=64, help='number of samples from generator')
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument("--personal_learning_rate", type=float, default=0.001,
                        help="Personalized learning rate to caculate theta aproximately using K steps")
    parser.add_argument("--ensemble_lr", type=float, default=1e-3, help="Ensemble learning rate.")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Average moving parameter for pFedMe, or Second learning rate of Per-FedAvg")
    parser.add_argument("--lamda", type=int, default=1, help="Regularization term")
    parser.add_argument("--mix_lambda", type=float, default=0.1, help="Mix lambda for FedMXI baseline")
    parser.add_argument("--embedding", type=int, default=0,
                        help="Use embedding layer in generator network")  # 在生成网络中使用嵌入层
    parser.add_argument("--num_glob_iters", type=int, default=200)
    # parser.add_argument("--num_glob_iters", type=int, default=20)
    parser.add_argument("--local_epochs", type=int, default=10)
    parser.add_argument("--num_users", type=int, default=3, help="Number of Users per round")
    parser.add_argument("--K", type=int, default=1, help="Computation steps")
    parser.add_argument("--times", type=int, default=1, help="running time")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="run device (cpu | cuda)")
    parser.add_argument("--result_path", type=str, default="results", help="directory path to save results")

    parser.add_argument("--n_epochs", type=int, default=50, help="Pre-training rounds for cgan")
    # parser.add_argument("--n_epochs", type=int, default=5, help="Pre-training rounds for cgan")

    parser.add_argument('--local_bs', type=int, default=100, help="local batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")

    parser.add_argument('--local_rep_ep', type=int, default=1,
                        help="number of local epoch for representation among local_ep")
    parser.add_argument('--reg_w', type=float, default=0.001, help="regularization of W ")
    parser.add_argument('--reg_reg_prior', type=float, default=0.001, help="regularization of W ")

    parser.add_argument('--model_type', type=str, default='reg', help="choosing the global model")
    parser.add_argument('--n_hidden', type=int, default=64, help="number of units in hidden layers")
    parser.add_argument('--dim_latent', type=int, default=64, help="latent dimension")
    parser.add_argument('--dim_in', type=int, default=64, help="latent dimension")
    parser.add_argument('--align_epochs', type=int, default=100, help="number of epochs for alignment during pretraining")
    # parser.add_argument('--align_epochs', type=int, default=10, help="number of epochs for alignment during pretraining")
    parser.add_argument('--align_epochs_altern', type=int, default=5,
                        help="number of epochs for alignment during alternate")
    parser.add_argument('--align_lr', type=float, default=0.0001, help="learning rate of alignment ")
    parser.add_argument('--align_bs', type=int, default=10, help="batch_size for alignment")
    parser.add_argument('--distance', type=str, default='wd', help="distance for alignment")

    parser.add_argument('--mean_target_variance', type=int, default=5, help="std of random prior means")
    parser.add_argument('--update_prior', type=int, default=1, help="updating prior (1 for True)")
    parser.add_argument('--update_net_preproc', type=int, default=1, help="updating preproc network")
    parser.add_argument('--start_optimize_rep', type=int, default=20, help="starting iterations for global model optim")

    parser.add_argument('--seed', type=int, default=0, help="choice of seed")
    parser.add_argument('--gpu', type=int, default=-1, help="gpu to use (if any")
    parser.add_argument('--num_workers', type=int, default=0, help="number of workers in dataloader")
    parser.add_argument('--test_freq', type=int, default=1, help="frequency of test eval")
    parser.add_argument('--timing', type=int, default=0, help="compute running time")
    parser.add_argument('--savedir', type=str, default='./save/', help="save dire")

    # 任务、组织、数据集相关参数
    parser.add_argument('--taskID', type=str, default='1713852468385451801')
    parser.add_argument('--test_dataset', type=str, default='test_dataset')
    parser.add_argument('--org_dataset_map', type=str,
                        default='{"org1": "dataset1", "org2": "dataset2", "org3": "dataset3"}')
    parser.add_argument('--accepted_orgs', type=str, default='["InvitedOrgs2", "InvitedOrgs3"]')
    args = parser.parse_args()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    print("=" * 80)
    print("Summary of training process:")
    print("Algorithm: {}".format(args.algorithm))
    print("Batch size: {}".format(args.batch_size))
    print("Learing rate       : {}".format(args.learning_rate))
    print("Ensemble learing rate       : {}".format(args.ensemble_lr))
    print("Average Moving       : {}".format(args.beta))
    print("Subset of users      : {}".format(args.num_users))
    print("Number of global rounds       : {}".format(args.num_glob_iters))
    print("Number of local rounds       : {}".format(args.local_epochs))
    print("Dataset       : {}".format(args.dataset))
    #    print("Local Model       : {}".format(args.model))
    print("Device            : {}".format(args.device))
    print("=" * 80)
    main(args)
