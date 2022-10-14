import argparse

from model.cnn import CNN_OriginalFedAvg
from model.cnn import CIFAR10_CNN
from model.cnn import MNIST_CNN
from model.cnn import Femnist_CNN
from model.cnn import CIFAR100_CNN

from data.femnist_certain.femnist import getFemnist_196clients
from data.cifar10.CIFAR10 import getCIFAR10
from data.cifar10.CIFAR10 import getCIFAR10_2label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_2label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_4label_each_100client
from data.cifar10.CIFAR10 import number_of_classes as cifar10_number_of_classes
from data.mnist.MNIST import getMNIST_2label_100client
from data.cifar100.CIFAR100 import getCIFAR100_100client

# from data.mnist.MNIST import getMNIST
# from data.mnist.MNIST import number_of_classes as number_of_classes
from algorithm.fedScaffold2.server import FedScaffold
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

    ret.epoch = 1
    ret.batch_mode = False

    ret.test_frequency = 1
    ret.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ret.lr_decay = 0.998
    return ret

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def run_mnist_2label_100client():
    args = get_args()
    data = getMNIST_2label_100client()
    model = MNIST_CNN()
    args.data = "MNIST"

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')

    args.comm_rounds = 200
    args.glr = 1
    for random_seed in range(5, 8):
        args.random_seed = random_seed
        run_name = "FedSa" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
        if (args.batch_mode):
            run_name = run_name + "batch"
        wandb.init(
            project="fedsurgery-mnist-100client_fullparticipation",
            name=run_name,
            config=args
        )
        set_random_seed(args.random_seed)
        server = FedScaffold(model, data, args)
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
    args.glr = 1
    args.smart_sample = False
    args.lr_decay = 1
    args.dropout = (0,0,0)
    # for number_of_clients_per_round, sample_num in zip([10, 20, 30, 40], [7, 10, 15, 20]):
    for number_of_clients_per_round, sample_num in zip([20], [15]):
        args.number_of_clients_per_round = number_of_clients_per_round
        args.greedy_mu = sample_num/number_of_clients_per_round + 0.01
        for epoch in range(1, 2):
            args.epoch = epoch
            for smart_sample in [False]:
                for random_seed in range(1, 2):
                    args.random_seed = random_seed
                    args.smart_sample = smart_sample  # none, init

                    run_name = "FedSa" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                    if (args.batch_mode):
                        run_name = run_name + "batch"
                    if (args.smart_sample):
                        run_name = 'smart_sample' + run_name
                    wandb.init(
                        project="fedSurg-CIFAR10-100-per-round-100-clients",
                        name=run_name,
                        config=args
                    )
                    set_random_seed(args.random_seed)
                    model = CIFAR10_CNN(args.dropout)
                    server = FedScaffold(model, data, args)
                    server.train()
                    wandb.finish()


def run_cifar10_2label_part_client():

    args = get_args()
    data = getCIFAR10_2label_each_100client()
    # data = getCIFAR10_2label_100client()
    model = CIFAR10_CNN((0, 0, 0))
    args.data = "CIFAR10"

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')

    args.comm_rounds = 300
    args.glr = 1
    args.opt = 2
    for random_seed in range(6, 9):
        args.random_seed = random_seed
        run_name = "FedSa" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
        if (args.batch_mode):
            run_name = run_name + "batch"
        wandb.init(
            project="fedSurg-CIFAR10-100-per-round-100-clients",
            name=run_name,
            config=args
        )
        set_random_seed(args.random_seed)
        server = FedScaffold(model, data, args)
        server.train()
        wandb.finish()
def run_femnist():
    args = get_args()
    data = getFemnist_196clients()
    model = Femnist_CNN()
    args.data = "femnist"

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')

    args.comm_rounds = 250
    args.number_of_clients = 196
    args.number_of_clients_per_round = 196
    args.glr = 1
    for random_seed in range(102, 104):
        args.random_seed = random_seed
        args.epoch = 1
        run_name = "fedSa" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
        if (args.batch_mode):
            run_name = run_name + "batch"
        wandb.init(
            project="fedsurgery-femnist-100client_fullparticipation",
            name=run_name,
            config=args
        )
        set_random_seed(args.random_seed)
        server = FedScaffold(model, data, args)
        server.train()
        wandb.finish()
def run_cifar100_2label_100client():

    args = get_args()
    data = getCIFAR100_100client()
    args.data = "CIFAR100"
    # args.epoch = 5
    args.lr = 0.05

    wandb.login(key='')



    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 600
    args.glr = 1
    # args.lr_decay = 0.995
    args.dropout = (0.3,0.3,0.3)
    args.smart_sample = False
    for rand_seed in range(0, 1):
        for smart_sample in [False]:
            args.smart_sample = smart_sample  # none, init
            for epoch in range(5, 6):
                args.random_seed = rand_seed
                args.epoch = epoch
                run_name = "fedSca" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                wandb.init(
                    project="fedSurg-CIFAR100-100-per-round-100-clients",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                model = CIFAR100_CNN(args.dropout)
                server = FedScaffold(model, data, args)
                server.train()
                wandb.finish()



def run_cifar10_4label_100client():

    args = get_args()
    args.smart_sample = False
    args.dropout = (0,0,0)
    data = getCIFAR10_4label_each_100client()
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
    args.comm_rounds = 500
    for epoch in range(1, 2):
        args.epoch = epoch
        for random_seed in range(0, 2):
            args.random_seed = random_seed
            run_name = 'Four_LABEL' + "fedSca" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
            if (args.batch_mode):
                run_name = run_name + "batch"
            wandb.init(
                project="4label-cifar10",
                name=run_name,
                config=args
            )
            set_random_seed(args.random_seed)
            model = CIFAR10_CNN(args.dropout)
            server = FedScaffold(model, data, args)
            server.train()
            wandb.finish()

if __name__ == '__main__':
    run_cifar10_2label_100client_cluster()
    # run_mnist_2label_100client()
    # run_cifar10_2label_100client()
    # run_cifar10_4label_100client()
    # run_cifar100_2label_100client()
    # run_femnist()
    # run_cifar10_4label_100client()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
