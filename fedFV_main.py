import argparse

from model.cnn import CNN_OriginalFedAvg
from model.cnn import CIFAR10_CNN
from model.cnn import MNIST_CNN
from model.cnn import CIFAR100_CNN
from model.cnn import Femnist_CNN
from model.cnn import AlexNet

from data.cifar10.CIFAR10 import getCIFAR10
from data.cifar10.CIFAR10 import getCIFAR10_2label_each_100client
from data.cifar10.CIFAR10 import number_of_classes as cifar10_number_of_classes
from data.cifar100.CIFAR100 import getCIFAR100_100client
from data.mnist.MNIST import getMNIST_2label_100client
from data.femnist_certain.femnist import getFemnist_196clients

# from data.mnist.MNIST import getMNIST
# from data.mnist.MNIST import number_of_classes as number_of_classes
from algorithm.fedSV.server import FedSV
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
    ret = argparse.Namespace
    # ret.comm_rounds = 500

    ret.number_of_clients = 100
    # ret.number_of_clients_per_round = 100

    ret.lr = 0.05
    ret.batchsize = 50
    # ret.random_seed = 3


    ret.sugery = True

    ret.epoch = 1
    ret.batch_mode = False
    ret.smart_sample = False

    ret.test_frequency = 1
    ret.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ret.weight_decay = 0
    ret.lr_decay = 1
    # ret.external_surgery = False

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
    model = MNIST_CNN()
    args.data = "MNIST"
    args.alpha = 0.5

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='a4e7793c50e34330c02de29fb678537bd57d078b')

    args.comm_rounds = 200
    for random_seed in range(5, 8):
        args.number_of_clients_per_round = 100
        args.random_seed = random_seed
        if (args.sugery != False):
            run_name = "FedSV" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.tau) + "tau"
        if (args.batch_mode):
            run_name = run_name + "batch"
        wandb.init(
            project="fedsurgery-mnist-100client_fullparticipation",
            name=run_name,
            config=args
        )
        set_random_seed(args.random_seed)
        server = FedSV(model, data, args)
        server.train()
        wandb.finish()


def run_cifar10_2label_100client_cluster():

    args = get_args()
    data = getCIFAR10_2label_each_100client()
    # data = getCIFAR10_2label_100client()
    args.data = "CIFAR10"
    args.dropout = (0,0,0)
    args.alpha = 0.5

    dataCounter = np.zeros([100, 10])
    for user_idx in range(100):
        labels = [y for (x, y) in data[0][user_idx]]
        a = Counter(labels)
        for t in range(10):
            dataCounter[user_idx][t] = a[t]
    np.save("cifar10_2label100client_cout.tb.npy", dataCounter)


    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='a4e7793c50e34330c02de29fb678537bd57d078b')



    args.number_of_clients = 100
    # args.number_of_clients_per_round = 20
    args.comm_rounds = 500
    args.lr_decay = 1
    args.tau = 1
    args.external_surgery = True

        # for number_of_clients_per_round in [10,20, 50]:
    # for number_of_clients_per_round, sample_num in zip([10, 20, 30, 40], [7, 15, 20, 30]):
    # for number_of_clients_per_round, sample_num in zip([10], [7]):
    for number_of_clients_per_round, sample_num in zip([20], [15]):
        args.number_of_clients_per_round = number_of_clients_per_round
        args.greedy_mu = sample_num/number_of_clients_per_round + 0.01
        for epoch in range(1, 2):
            args.epoch = epoch
            for smart_sample in [False]:
                for random_seed in range(1):
                    args.smart_sample = smart_sample  # none, init
                    args.random_seed = random_seed
                    if (args.sugery != False):
                        run_name = "FedFV" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.tau) + "tau"
                    if (args.batch_mode):
                        run_name = run_name + "batch"
                    if (args.smart_sample):
                        run_name = 'smart_sample' + run_name
                    wandb.init(
                        project= "fedSurg-CIFAR10-100-per-round-100-clients",
                        name=run_name,
                        config=args
                    )
                    set_random_seed(args.random_seed)
                    model = CIFAR10_CNN(args.dropout)
                    server = FedSV(model, data, args)
                    server.train()
                    wandb.finish()


def run_cifar10_2label_100client():

    args = get_args()
    data = getCIFAR10_2label_each_100client()
    # data = getCIFAR10_2label_100client()
    args.data = "CIFAR10"
    args.alpha = 0.5

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='a4e7793c50e34330c02de29fb678537bd57d078b')

    args.tau = 1
    args.comm_rounds = 500
    for number_of_clients_per_round in [100]:
        for random_seed in range(0, 2):
            # for smart_sample in [True, False]:
            for smart_sample in [False]:
            # for smart_sample in [True, False]:
                # for number_of_clients_per_round in [100]:
                # for number_of_clients_per_round in [30, 40]:
                    args.number_of_clients_per_round = number_of_clients_per_round

                    args.smart_sample = smart_sample
                    args.random_seed = random_seed
                    args.external_surgery = True

                    if (args.sugery != False):
                        run_name = "FedFV" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.tau) + "tau"
                    if (args.batch_mode):
                        run_name = run_name + "batch"
                    if args.smart_sample:
                        run_name = "smart_sample" + run_name
                    else:
                        run_name = "none" + run_name
                    wandb.init(
                        project="fedSurg-CIFAR10-100-per-round-100-clients",
                        name=run_name,
                        config=args
                    )
                    set_random_seed(args.random_seed)
                    model = CIFAR10_CNN((0, 0, 0))
                    server = FedSV(model, data, args)
                    server.train()
                    wandb.finish()

def run_cifar100_2label_100client():

    args = get_args()
    data = getCIFAR100_100client()
    args.data = "CIFAR100"
    args.alpha = 0.5
    # args.epoch = 5

    wandb.login(key='a4e7793c50e34330c02de29fb678537bd57d078b')

    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 600
    for rand_seed in range(1):
        for epoch in range(5, 6):
            args.random_seed = rand_seed
            args.epoch = epoch
            args.lr = 0.05
            args.external_surgery = False
            if (args.sugery):
                run_name =  "FedSV"+ str(args.lr) + "lr" + str(args.epoch) + "epoch"
            else:
                run_name = "fedAvg" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
            if (args.batch_mode):
                run_name = run_name + "batch"
            wandb.init(
                project="fedSurg-CIFAR100-100-per-round-100-clients",
                name=run_name,
                config=args
            )
            set_random_seed(args.random_seed)
            # model = AlexNet()
            model = CIFAR100_CNN((0, 0, 0))
            server = FedSV(model, data, args)
            server.train()
            wandb.finish()

def run_femnist():
    args = get_args()
    data = getFemnist_196clients()
    model = Femnist_CNN()
    args.data = "femnist"
    args.alpha = 0.5

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='a4e7793c50e34330c02de29fb678537bd57d078b')

    args.comm_rounds = 250
    args.number_of_clients = 196
    args.number_of_clients_per_round = 196
    for random_seed in range(101, 104):
        args.random_seed = random_seed
        args.epoch = 1
        if (args.sugery != False):
            run_name = "FedFV" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
        else:
            run_name = "fedAvg" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
        if (args.batch_mode):
            run_name = run_name + "batch"
        wandb.init(
            project="fedsurgery-femnist-100client_fullparticipation",
            name=run_name,
            config=args
        )
        set_random_seed(args.random_seed)
        server = FedSV(model, data, args)
        server.train()
        wandb.finish()

if __name__ == '__main__':
    # run_mnist_2label_100client()
    # run_cifar10_2label_100client()
    run_cifar10_2label_100client_cluster()
    # run_cifar100_2label_100client()
    # run_femnist()
    # args = get_args()
    # set_random_seed(args.random_seed)
    # data = getCIFAR10_2label_each_100client()
    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    # wandb.login(key='a4e7793c50e34330c02de29fb678537bd57d078b')
    # if (args.sugery != "none"):
    #     run_name = args.sugery + str(args.lr) + "lr" +  str(args.alpha) + "a"
    # else:
    #     run_name = "fedAvg" + str(args.lr) + "lr" +  str(args.alpha) + "a"
    # if (args.batch_mode):
    #     run_name = run_name + "batch"
    # wandb.init(
    #     project="fedSurg",
    #     name=run_name,
    #     config=args
    # )
    #
    # server = FedSurg(args.model, data, args)
    # server.train()
    # wandb.finish()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
