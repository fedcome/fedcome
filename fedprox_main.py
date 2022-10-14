import argparse

from model.cnn import CNN_OriginalFedAvg
from model.cnn import CIFAR10_CNN
from model.cnn import MNIST_CNN

from model.cnn import CIFAR100_CNN
from data.cifar100.CIFAR100 import getCIFAR100_100client
from data.cifar10.CIFAR10 import getCIFAR10
from data.cifar10.CIFAR10 import getCIFAR10_2label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_2label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_3label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_4label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_5label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_7label_each_100client
from data.cifar10.CIFAR10 import number_of_classes as cifar10_number_of_classes
from data.mnist.MNIST import getMNIST_2label_100client


from data.femnist_certain.femnist import getFemnist_196clients
from model.cnn import Femnist_CNN

# from data.mnist.MNIST import getMNIST
# from data.mnist.MNIST import number_of_classes as number_of_classes
from algorithm.fedProx.server import FedProx
import torch
import wandb
import random
from collections import Counter


from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import numpy as np

ROOT = './'
RAW_DATA_PATH = ROOT + 'data/raw_data/'

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "./")))

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
def get_args():
    ret = argparse.Namespace()
    # ret.comm_rounds = 500

    ret.number_of_clients = 100
    ret.number_of_clients_per_round = 100

    ret.lr = 0.05
    ret.batchsize = 50
    ret.random_seed = 3


    # ret.mu = 2 # mu/2

    ret.epoch = 1
    ret.batch_mode = False

    ret.test_frequency = 1
    ret.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ret.lr_decay = 1

    # ret.model = CIFAR10_CNN((0, 0, 0))
    # ret.data = "MNIST"
    # ret.model = CNN_OriginalFedAvg([3, 32, 32], 10)
    # ret.model = MNIST_CNN()
    return ret

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def run_mnist_2label_100client():
    args = get_args()
    data = getMNIST_2label_100client()
    args.data = "MNIST"

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')

    args.comm_rounds = 200
    for random_seed in range(5, 8):
        args.random_seed = random_seed
        # for mu in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2]:
        for mu in [0.001]:
            args.mu = mu
            run_name = "Fedprox" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.mu) + "mu"
            wandb.init(
                project="fedsurgery-mnist-100client_fullparticipation",
                name=run_name,
                config=args
            )
            set_random_seed(args.random_seed)
            model = MNIST_CNN()
            server = FedProx(model, data, args)
            server.train()
            wandb.finish()

def run_femnist_2label_100client():
    args = get_args()
    data = getFemnist_196clients()
    model = Femnist_CNN()
    args.data = "Femnist"
    args.number_of_clients = 196
    args.number_of_clients_per_round = 196

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')

    args.comm_rounds = 250
    for random_seed in range(5, 8):
        args.random_seed = random_seed
        for mu in [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2]:
            args.mu = mu
            run_name = "Fedprox" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.mu) + "mu"
            wandb.init(
                project="fedsurgery-femnist-100client_fullparticipation",
                name=run_name,
                config=args
            )
            set_random_seed(args.random_seed)
            server = FedProx(model, data, args)
            server.train()
            wandb.finish()
def run_cifar10_2label_100client_cluster():

    args = get_args()
    data = getCIFAR10_2label_each_100client()
    # data = getCIFAR10_2label_100client()
    args.data = "CIFAR10"

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')

    args.comm_rounds = 500
    args.dropout = (0,0,0)
    for number_of_clients_per_round, sample_num in zip([10, 20, 30, 40], [7, 10, 15, 20]):
        args.number_of_clients_per_round = number_of_clients_per_round
        args.greedy_mu = sample_num/number_of_clients_per_round + 0.01
        for epoch in range(1, 2):
            args.epoch = epoch
            for mu in [0.0001, 0]:
                for smart_sample in [True]:
                    for random_seed in range(2):
                        args.random_seed = random_seed
                        args.mu = mu
                        args.smart_sample = smart_sample  # none, init

                        run_name = "Fedprox" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.mu) + "mu"
                        if (args.smart_sample):
                            run_name = 'smart_sample' + run_name
                        wandb.init(
                            project= "fedSurg-CIFAR10-100-per-round-100-clients",
                            name=run_name,
                            config=args
                        )
                        set_random_seed(args.random_seed)
                        model = CIFAR10_CNN(args.dropout)
                        server = FedProx(model, data, args)
                        server.train()
                        wandb.finish()

def run_cifar10_2label_100client():

    args = get_args()
    data = getCIFAR10_2label_each_100client()
    # data = getCIFAR10_2label_100client()
    args.data = "CIFAR10"

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')

    args.smart_sample = False
    args.comm_rounds = 500
    args.dropout = (0,0,0)
    for mu in [0.0001, 0]:
        for number_of_clients_per_round in [10, 20, 30, 40]:
        # for mu in [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 2]:
            for random_seed in range(0, 1):
                args.number_of_clients_per_round = number_of_clients_per_round
                args.mu = mu
                args.random_seed = random_seed
                run_name = "Fedprox" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.mu) + "mu"
                wandb.init(
                    project="fedSurg-CIFAR10-100-per-round-100-clients",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                model = CIFAR10_CNN(args.dropout)
                server = FedProx(model, data, args)
                server.train()
                wandb.finish()

def run_cifar100_2label_100client():

    args = get_args()
    data = getCIFAR100_100client()
    args.data = "CIFAR100"
    # args.epoch = 5
    args.lr = 0.05
    args.smart_sample = False

    wandb.login(key='')



    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 600
    for rand_seed in range(4, 5):
        for epoch in range(5, 6):
            # for mu in [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 2]:
            for mu in [0]:
                args.mu = mu
                args.random_seed = rand_seed
                args.epoch = epoch
                run_name = "fedProx" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.mu) + "mu"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                wandb.init(
                    project="fedSurg-CIFAR100-100-per-round-100-clients",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                model = CIFAR100_CNN((0.1, 0.1, 0.1))
                server = FedProx(model, data, args)
                server.train()
                wandb.finish()
def run_cifar10_3label_100client():

    args = get_args()
    data = getCIFAR10_3label_each_100client()
    for user in data[0]:
        labels = [y for (x, y) in user]
        print(Counter(labels))
    # data = getCIFAR10_2label_100client()
    args.data = "CIFAR10"

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')

    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 600
    for epoch in range(1, 2):
        args.epoch = epoch
        for mu in [0.001, 0]:
            args.mu = mu
            for random_seed in range(0, 2):
                args.random_seed = random_seed
                run_name = "Three_LABEL" + "Fedprox" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.mu) + "mu"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                wandb.init(
                    project="fedSurg-CIFAR10-100-per-round-100-clients",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                model = CIFAR10_CNN((0, 0, 0))
                server = FedProx(model, data, args)
                server.train()
                wandb.finish()

def run_cifar10_4label_100client():

    args = get_args()
    data = getCIFAR10_4label_each_100client()
    for user in data[0]:
        labels = [y for (x, y) in user]
        print(Counter(labels))
    args.data = "CIFAR10"

    wandb.login(key='')

    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 500
    args.smart_sample = False
    for epoch in range(1, 2):
        args.epoch = epoch
        for mu in [0.001, 0]:
            args.mu = mu
            for random_seed in range(0, 2):
                args.random_seed = random_seed
                run_name = "Four_LABEL" + "Fedprox" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.mu) + "mu"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                wandb.init(
                    project="4label-cifar10",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                model = CIFAR10_CNN((0, 0, 0))
                server = FedProx(model, data, args)
                server.train()
                wandb.finish()

def run_cifar10_5label_100client():
    args = get_args()
    data = getCIFAR10_5label_each_100client()
    for user in data[0]:
        labels = [y for (x, y) in user]
        print(Counter(labels))
    # data = getCIFAR10_2label_100client()
    model = CIFAR10_CNN((0, 0, 0))
    args.data = "CIFAR10"

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')

    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 300
    for epoch in range(1, 2):
        args.epoch = epoch
        for mu in [0.000]:
            args.mu = mu
            for random_seed in range(0, 3):
                args.random_seed = random_seed
                run_name = "Five_LABEL" + "Fedprox" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.mu) + "mu"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                wandb.init(
                    project="fedSurg-CIFAR10-100-per-round-100-clients",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                server = FedProx(model, data, args)
                server.train()
                wandb.finish()

def run_cifar10_7label_100client():
    args = get_args()
    data = getCIFAR10_7label_each_100client()
    for user in data[0]:
        labels = [y for (x, y) in user]
        print(Counter(labels))
    # data = getCIFAR10_2label_100client()
    model = CIFAR10_CNN((0.1, 0.1, 0.1))
    args.data = "CIFAR10"

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')


    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 300
    for epoch in range(1, 2):
        args.epoch = epoch
        for mu in [0.001]:
            args.mu = mu
            for random_seed in range(0, 3):
                args.random_seed = random_seed
                run_name = "Seven_LABEL" + "Fedprox" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.mu) + "mu"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                wandb.init(
                    project="fedSurg-CIFAR10-100-per-round-100-clients",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                server = FedProx(model, data, args)
                server.train()
                wandb.finish()

if __name__ == '__main__':
    # run_mnist_2label_100client()
    # run_cifar10_4label_100client()
    # run_cifar10_3label_100client()
    # run_cifar10_2label_100client()
    run_cifar10_2label_100client_cluster()
    # run_cifar10_5label_100client()
    # run_cifar10_7label_100client()
    # run_femnist_2label_100client()
    # run_cifar100_2label_100client()