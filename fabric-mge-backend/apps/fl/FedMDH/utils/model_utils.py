import json
import torch
import os
from apps.fl.FL_for_matdata.utils import getcsv_mongo
from apps.fl.FedMDH.FLAlgorithms.trainmodel.models1 import Net2
from torch.utils.data import DataLoader
from apps.fl.FedMDH.FLAlgorithms.trainmodel.generator import Generator, Discriminator
from apps.fl.FedMDH.utils.model_config import *

METRICS = ['glob_mse', 'per_mse', 'glob_loss', 'glob_r2', 'per_loss', 'user_train_time', 'server_agg_time']


def get_data_dir(dataset, taskID):
    if 'Material' in dataset:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)

        path_prefix = os.path.join(parent_dir, 'data', 'Material')
        train_data_dir = os.path.join(path_prefix, 'train')
        test_data_dir = os.path.join(path_prefix, 'test')
        os.makedirs(train_data_dir, exist_ok=True)
        os.makedirs(test_data_dir, exist_ok=True)
        proxy_data_dir = 'data/proxy_data/mat/'
    else:
        raise ValueError("Dataset not recognized.")
    return train_data_dir, test_data_dir, proxy_data_dir


def create_ptfile(data_dir, cfile_name, args):
    org_dataset_map = json.loads(args.org_dataset_map)
    file_paths = []
    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path):
            file_paths.append(file_path)

    data_dict = {'users': [], 'user_data': {}, 'num_samples': []}

    user_id_format = 'f_{:05d}'
    for user_name, dataset_name in org_dataset_map.items():
        user = user_name  
        if cfile_name == "train":
            df = getcsv_mongo(dataset_name)
        else:
            df = getcsv_mongo(args.test_dataset)

        y = torch.tensor(df['e_form'].values)
        x = torch.tensor(df.drop(columns=['e_form']).select_dtypes(include=[float, int]).values,
                         dtype=torch.float32)
        data_dict['users'].append(user)
        data_dict['user_data'][user] = {'x': x, 'y': y}
        data_dict['num_samples'].append(len(df))

    output_file = os.path.join(data_dir, f"{cfile_name}.pt")  
    torch.save(data_dict, output_file)


def read_data(dataset, args):

    train_data_dir, test_data_dir, proxy_data_dir = get_data_dir(dataset, args.taskID)
    if not any(fname.endswith('.pt') for fname in os.listdir(train_data_dir)):
        create_ptfile(train_data_dir, cfile_name='train', args=args)
    else:
        print(f"Skipped create_ptfile for train_data_dir because .pt file already exists.")

    if not any(fname.endswith('.pt') for fname in os.listdir(test_data_dir)):
        create_ptfile(test_data_dir, cfile_name='test', args=args)
    else:
        print(f"Skipped create_ptfile for test_data_dir because .pt file already exists.")
    # create_ptfile(test_data_dir,file_name='test')
    clients = []
    groups = []
    train_data = {}
    test_data = {}
    proxy_data = {}

    train_files = os.listdir(train_data_dir)
    train_files = [f for f in train_files if f.endswith('.json') or f.endswith(".pt")]
    for f in train_files:
        file_path = os.path.join(train_data_dir, f)
        if file_path.endswith("json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        elif file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)  #
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))

        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        train_data.update(cdata['user_data'])

    clients = list(sorted(train_data.keys()))

    test_files = os.listdir(test_data_dir)
    test_files = [f for f in test_files if f.endswith('.json') or f.endswith(".pt")]
    for f in test_files:
        file_path = os.path.join(test_data_dir, f)
        if file_path.endswith(".pt"):
            with open(file_path, 'rb') as inf:
                cdata = torch.load(inf)
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as inf:
                cdata = json.load(inf)
        else:
            raise TypeError("Data format not recognized: {}".format(file_path))
        test_data.update(cdata['user_data'])

    if proxy_data_dir and os.path.exists(proxy_data_dir):
        proxy_files = os.listdir(proxy_data_dir)
        proxy_files = [f for f in proxy_files if f.endswith('.json') or f.endswith(".pt")]
        for f in proxy_files:
            file_path = os.path.join(proxy_data_dir, f)
            if file_path.endswith(".pt"):
                with open(file_path, 'rb') as inf:
                    cdata = torch.load(inf)
            elif file_path.endswith(".json"):
                with open(file_path, 'r') as inf:
                    cdata = json.load(inf)
            else:
                raise TypeError("Data format not recognized: {}".format(file_path))
            proxy_data.update(cdata['user_data'])

    return clients, groups, train_data, test_data, proxy_data


def read_proxy_data(proxy_data, dataset, batch_size):
    X, y = proxy_data['x'], proxy_data['y']
    X, y = convert_data(X, y, dataset=dataset)
    dataset = [(x, y) for x, y in zip(X, y)]
    proxyloader = DataLoader(dataset, batch_size, shuffle=True)
    iter_proxyloader = iter(proxyloader)
    return proxyloader, iter_proxyloader


def aggregate_data_(clients, dataset, dataset_name, batch_size):
    combined = []
    unique_labels = []
    for i in range(len(dataset)):
        id = clients[i]
        user_data = dataset[id]
        X, y = convert_data(user_data['x'], user_data['y'], dataset=dataset_name)
        combined += [(x, y) for x, y in zip(X, y)]
        # unique_y=torch.unique(y)
        # unique_y = unique_y.detach().numpy()
        # unique_labels += list(unique_y)

    data_loader = DataLoader(combined, batch_size, shuffle=True)
    iter_loader = iter(data_loader)
    return data_loader, iter_loader


def aggregate_user_test_data(data, dataset_name, batch_size):
    clients, loaded_data = data[0], data[3]
    data_loader, _ = aggregate_data_(clients, loaded_data, dataset_name, batch_size)
    return data_loader


def aggregate_user_data(data, dataset_name, batch_size):
    # data contains: clients, groups, train_data, test_data, proxy_data
    clients, loaded_data = data[0], data[2]
    data_loader, data_iter = aggregate_data_(clients, loaded_data, dataset_name, batch_size)
    return data_loader, data_iter


def convert_data(X, y, dataset=''):
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X).type(torch.float32)  
        y = torch.Tensor(y).type(torch.float32)  
    return X, y


def read_user_data(index, data, dataset=''):
    # data contains: clients, groups, train_data, test_data, proxy_data(optional)
    id = data[0][index]
    train_data = data[2][id]
    test_data = data[3][id]
    X_train, y_train = convert_data(train_data['x'], train_data['y'], dataset=dataset)
    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    X_test, y_test = convert_data(test_data['x'], test_data['y'], dataset=dataset)
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return id, train_data, test_data


def get_dataset_name(dataset):
    dataset = dataset.lower()
    passed_dataset = dataset.lower()
    if 'material' in dataset:
        passed_dataset = 'material'
    else:
        raise ValueError('Unsupported dataset {}'.format(dataset))
    return passed_dataset


def create_generative_model(dataset, algorithm='', embedding=False):
    passed_dataset = get_dataset_name(dataset)
    assert any([alg in algorithm for alg in ['FedMDH']])
    if 'FedMDH' in algorithm:
        # temporary roundabout to figure out the sensitivity of the generator network & sampling size
        if 'cnn' in algorithm:
            gen_model = algorithm.split('-')[1]
            passed_dataset += '-' + gen_model
        elif '-MDH' in algorithm:  # we use more lightweight network for sensitivity analysis
            passed_dataset += '-cnn1'
    return Generator(passed_dataset, embedding=embedding, latent_layer_idx=0)


def create_discriminator_model(dataset, algorithm='', embedding=False):
    passed_dataset = get_dataset_name(dataset)
    assert any([alg in algorithm for alg in ['FedMDH']])
    if 'FedMDH' in algorithm:
        # temporary roundabout to figure out the sensitivity of the generator network & sampling size
        if 'cnn' in algorithm:
            gen_model = algorithm.split('-')[1]
            passed_dataset += '-' + gen_model
        elif '-MDH' in algorithm:  # we use more lightweight network for sensitivity analysis
            passed_dataset += '-cnn1'
    return Discriminator(passed_dataset, embedding=embedding)


def create_model(input_size):
    # passed_dataset = get_dataset_name(dataset)#改成MLP
    model = Net2(input_size=input_size)
    return model


def polyak_move(params, target_params, ratio=0.1):
    for param, target_param in zip(params, target_params):
        param.data = param.data - ratio * (param.clone().detach().data - target_param.clone().detach().data)


def meta_move(params, target_params, ratio):
    for param, target_param in zip(params, target_params):
        target_param.data = param.clone().data + ratio * (target_param.clone().data - param.clone().data)


def moreau_loss(params, reg_params):
    # return 1/T \sum_i^T |param_i - reg_param_i|^2
    losses = []
    for param, reg_param in zip(params, reg_params):
        losses.append(torch.mean(torch.square(param - reg_param.clone().detach())))
    loss = torch.mean(torch.stack(losses))
    return loss


def l2_loss(params):
    losses = []
    for param in params:
        losses.append(torch.mean(torch.square(param)))
    loss = torch.mean(torch.stack(losses))
    return loss


def update_fast_params(fast_weights, grads, lr, allow_unused=False):
    """
    Update fast_weights by applying grads.
    :param fast_weights: list of parameters.
    :param grads: list of gradients
    :param lr:
    :return: updated fast_weights .
    """
    for grad, fast_weight in zip(grads, fast_weights):
        if allow_unused and grad is None: continue
        grad = torch.clamp(grad, -10, 10)
        fast_weight.data = fast_weight.data.clone() - lr * grad
    return fast_weights


def init_named_params(model, keywords=['encode']):
    named_params = {}
    # named_params_list = []
    for name, params in model.named_layers.items():
        if any([key in name for key in keywords]):
            named_params[name] = [param.clone().detach().requires_grad_(True) for param in params]
            # named_params_list += named_params[name]
    return named_params  # , named_params_list


def get_log_path(args, algorithm, seed, gen_batch_size=32):
    alg = args.dataset + "_" + algorithm
    alg += "_" + str(args.learning_rate) + "_" + str(args.num_users)
    alg += "u" + "_" + str(args.batch_size) + "b" + "_" + str(args.local_epochs)
    alg = alg + "_" + str(seed)
    if 'FedMDH' in algorithm:  # to accompany experiments for author rebuttal
        alg += "_embed" + str(args.embedding)
        if int(gen_batch_size) != int(args.batch_size):
            alg += "_gb" + str(gen_batch_size)
    return alg