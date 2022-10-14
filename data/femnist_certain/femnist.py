import os
import torch
import torchvision
import random
import numpy as np
import json
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
# from utils.tools import read_dir, setup_seed

def read_dir(data_dir):
    clients = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        data.update(cdata['user_data'])

    # clients = list(sorted(data.keys()))
    return clients, data


class FEMNIST_DATASET(Dataset):
    def __init__(self, data, labels, transform=None):
        self._data = data
        self._labels = labels
        self._transform = transform

    def __getitem__(self, item):
        _x = self._data[item]
        _y = self._labels[item]
        assert len(_x) == 784
        _x = np.array(_x)
        _x = _x.reshape((1, 28, 28))
        if self._transform is not None:
            _x = self._transform(_x)
        else:
            _x = torch.tensor(_x)
        _x = _x.reshape((1, 28, 28))
        _x = _x.float()
        _y = torch.tensor(_y).long()
        return _x, _y

    def __len__(self):
        return len(self._labels)

def _get_femnist_data(use='train', transform=None):
    femnist_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(femnist_dir, 'train')
    test_path = os.path.join(femnist_dir, 'test')
    train_clients, train_dataset = read_dir(train_path)
    test_clients, test_dataset = read_dir(test_path)

    if use == 'train':
        all_dataset = train_dataset
        all_clients = train_clients
    else:
        all_dataset = test_dataset
        all_clients = test_clients

    data2user = []
    for client in all_clients:
        data = all_dataset[client]['x']
        labels = all_dataset[client]['y']

        dataset = FEMNIST_DATASET(data=data, labels=labels, transform=transform)
        # data_loader = DataLoader(dataset=dataset, batch_size=new_batch_size, shuffle=True, num_workers=0)

        data2user.append([(x, y) for x, y in dataset])

    return data2user

def _get_femnist_dataLoaders(use='train', batch_size=10, transform=None):
    femnist_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(femnist_dir, 'train')
    test_path = os.path.join(femnist_dir, 'test')
    train_clients, train_dataset = read_dir(train_path)
    test_clients, test_dataset = read_dir(test_path)

    if use == 'train':
        all_dataset = train_dataset
        all_clients = train_clients
    else:
        all_dataset = test_dataset
        all_clients = test_clients

    dataLoaders = {}
    for client in all_clients:
        data = all_dataset[client]['x']
        labels = all_dataset[client]['y']

        new_batch_size = min(batch_size, len(labels))

        dataset = FEMNIST_DATASET(data=data, labels=labels, transform=transform)
        data_loader = DataLoader(dataset=dataset, batch_size=new_batch_size, shuffle=True, num_workers=0)

        dataLoaders[client] = data_loader

    return all_clients, dataLoaders


def get_femnist_dataLoaders(batch_size=50, train_transform=None, test_transform=None):
    # setup_seed(rs=24)
    train_all_clients, trainLoaders = _get_femnist_dataLoaders(use='train', batch_size=batch_size,
                                                               transform=train_transform)
    test_all_clients, testLoaders = _get_femnist_dataLoaders(use='test', batch_size=batch_size,
                                                             transform=test_transform)
    train_all_clients.sort()
    test_all_clients.sort()
    assert train_all_clients == test_all_clients
    return train_all_clients, trainLoaders, testLoaders

def getFemnist_196clients():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    user2data_train = _get_femnist_data('train', transform)
    user2data_test = _get_femnist_data('test', transform)
    return user2data_train, user2data_test


if __name__ == '__main__':
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    user2data_train = _get_femnist_data('train', transform)
    user2data_test = _get_femnist_data('test', transform)
    print("ok")
    # clients, _trainLoaders, _testLoaders = get_femnist_dataLoaders()
    # for _, (data, labels) in enumerate(_trainLoaders[clients[5]]):
    #     print(labels)
    #
    # print("============")

    # for _, (data, labels) in enumerate(_testLoaders[clients[5]]):
    #     print(labels)

