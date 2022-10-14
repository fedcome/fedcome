import argparse

from model.cnn import CNN_OriginalFedAvg
from model.cnn import Femnist_CNN
from model.cnn import CIFAR100_CNN
from model.cnn import CIFAR10_CNN
# from model.cnn import CIFAR10_CNN
# from model.reparam_networks import MNIST_CNN
# from model.reparam_networks import CIFAR10_CNN
from model.cnn import MNIST_CNN
from data.cifar10.CIFAR10 import getCIFAR10
from data.cifar10.CIFAR10 import getCIFAR10_2label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_2label_each_100client
from data.cifar10.CIFAR10 import number_of_classes as cifar10_number_of_classes
from data.cifar10.CIFAR10 import getCIFAR10_3label_each_100client, getCIFAR10_4label_each_100client, getCIFAR10_5label_each_100client, getCIFAR10_7label_each_100client

from data.cifar100.CIFAR100 import getCIFAR100_100client
from data.femnist_certain.femnist import getFemnist_196clients
from data.mnist.MNIST import getMNIST_2label_100client

# from data.mnist.MNIST import getMNIST
# from data.mnist.MNIST import number_of_classes as number_of_classes
from algorithm.fedDyn.server import FedDyn
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

    ret.number_of_clients = 100
    ret.number_of_clients_per_round = 100

    ret.lr = 0.05
    ret.batchsize = 50

    ret.epoch = 1
    ret.batch_mode = False

    ret.test_frequency = 1
    ret.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ret.alpha_coef = 0.01
    ret.lr_decay = 0.998
    # ret.lr_decay = 1
    ret.weight_decay = 0
    ret.smart_sample = False 
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
    for alpha in [0.01]:
        args.alpha_coef = alpha
        # args.random_seed = random_seed
        run_name = "FedDyn_" + str(args.alpha)
        if (args.batch_mode):
            run_name = run_name + "batch" + str(args.alpha_coef) + "alpha"
        wandb.init(
            project="fedsurgery-mnist-100client_fullparticipation",
            name=run_name,
            config=args
        )
        set_random_seed(args.random_seed)
        model = MNIST_CNN()
        server = FedDyn(model, data, args)
        server.train()
        wandb.finish()
def run_femnist():
    args = get_args()
    data = getFemnist_196clients()
    args.data = "femnist"

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()

    args.comm_rounds = 250
    # args.comm_rounds = 50
    args.number_of_clients = 196
    args.number_of_clients_per_round = 196
    args.glr = 1
    args.alpha_coef = 1
    for alpha in [0.01, 0.1]:
        args.alpha_coef = alpha
        for random_seed in range(102, 104):
            args.random_seed = random_seed
            args.epoch = 1
            run_name = "FedDyn_" + str(args.alpha_coef)+"alpha"
            if (args.batch_mode):
                run_name = run_name + "batch"
            wandb.init(
                project="fedsurgery-femnist-100client_fullparticipation",
                name=run_name,
                config=args
            )
            set_random_seed(args.random_seed)
            model = Femnist_CNN()
            server = FedDyn(model, data, args)
            server.train()
            wandb.finish()

def run_cifar10_2label_100client():

    args = get_args()
    data = getCIFAR10_2label_each_100client()
    # data = getCIFAR10_2label_100client()
    # args.dropout = (0, 0, 0)
    args.data = "CIFAR10"

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='a4e7793c50e34330c02de29fb678537bd57d078b')

    args.comm_rounds = 500
    # for alpha in [0.01, 0.1, 1]:
    # for alpha in [0.01, 0.1, 1]:
    args.dropout = (0.3,0.3,0.3)
    for alpha in [0.01]:
        args.alpha_coef = alpha
        run_name = "fedDYN" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.alpha_coef)+"alpha"
        # args.random_seed = random_seed
        for random_seed in range(0,1):
            args.run_name = run_name
            args.random_seed = random_seed
            if (args.batch_mode):
                run_name = run_name + "batch" + str(args.alpha_coef) + str(alpha)
            wandb.init(
                project="fedsurgcifar10_100client-500Round",
                name=run_name,
                config=args
            )

            set_random_seed(args.random_seed)
            model = CIFAR10_CNN(args.dropout)
            server = FedDyn(model, data, args)
            server.train()
            wandb.finish()
def run_cifar10_2label_100client_cluster():

    args = get_args()
    data = getCIFAR10_2label_each_100client()
    # data = getCIFAR10_2label_100client()
    args.data = "CIFAR10"

    dataCounter = np.zeros([100, 10])
    for user_idx in range(100):
        labels = [y for (x, y) in data[0][user_idx]]
        a = Counter(labels)
        for t in range(10):
            dataCounter[user_idx][t] = a[t]
    np.save("cifar10_2label100client_cout.tb.npy", dataCounter)


    wandb.login(key='')



    args.number_of_clients = 100
    args.comm_rounds = 500
    args.lr_decay = 1
    for number_of_clients_per_round, sample_num in zip([10], [7]):
        args.number_of_clients_per_round = number_of_clients_per_round
        args.greedy_mu = sample_num/number_of_clients_per_round + 0.01
        for epoch in range(1, 2):
            args.epoch = epoch
            for smart_sample in [False, True]:
                for random_seed in range(2):
                    for dropout_prob in [0.3]:
                        args.smart_sample = smart_sample  # none, init
                        args.random_seed = random_seed
                        args.dropout = (dropout_prob, dropout_prob, dropout_prob)
                        run_name = "fedDYN" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.alpha_coef)+"alpha"+str(dropout_prob)+'dropout'
                        if (args.smart_sample):
                            run_name = 'smart_sample' + run_name
                        wandb.init(
                            project= "fedSurg-CIFAR10-100-per-round-100-clients",
                            name=run_name,
                            config=args
                        )
                        set_random_seed(args.random_seed)
                        model = CIFAR10_CNN(args.dropout)
                        server = FedDyn(model, data, args)
                        server.train()
                        wandb.finish()

def run_cifar100_2label_100client():

    args = get_args()
    data = getCIFAR100_100client()
    args.data = "CIFAR100"
    # args.epoch = 5
    args.lr = 0.05

    wandb.login(key='a4e7793c50e34330c02de29fb678537bd57d078b')



    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 600
    # args.comm_rounds = 50
    args.glr = 1
    args.weight_decay = 0
    args.smart_sample = False
    # args.lr_decay = 0.995
    for alpha_coef in [0.01]:
        for epoch in range(5, 6):
            # for dropout_prob in [0.3, 0.6]:
            for dropout_prob in [0.6]:
                for rand_seed in range(1, 3):
                    args.random_seed = rand_seed
                    args.epoch = epoch
                    args.alpha_coef = alpha_coef
                    args.dropout = (dropout_prob, dropout_prob, dropout_prob)
                    run_name = "fedDYN" + str(args.lr) + "lr" + str(args.epoch) + "epoch" + str(args.alpha_coef)+"alpha"+str(dropout_prob)+'dropout'
                    if (args.batch_mode):
                        run_name = run_name + "batch"
                    wandb.init(
                        project="fedSurg-CIFAR100-100-per-round-100-clients",
                        name=run_name,
                        config=args
                    )
                    set_random_seed(args.random_seed)
                    model = CIFAR100_CNN(args.dropout)
                    server = FedDyn(model, data, args)
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
    wandb.login(key='a4e7793c50e34330c02de29fb678537bd57d078b')
    args.dropout = (0.3, 0.3, 0.3)



    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 500
    for epoch in range(1, 2):
        args.epoch = epoch
        for random_seed in range(0, 2):
            args.random_seed = random_seed
            run_name = 'Three_LABEL' + "fedDYN" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
            if (args.batch_mode):
                run_name = run_name + "batch"
            wandb.init(
                project="3label-cifar10",
                name=run_name,
                config=args
            )
            set_random_seed(args.random_seed)
            model = CIFAR10_CNN(args.dropout)
            server = FedDyn(model, data, args)
            server.train()
            wandb.finish()

def run_cifar10_4label_100client():

    args = get_args()
    data = getCIFAR10_4label_each_100client()
    for user in data[0]:
        labels = [y for (x, y) in user]
        print(Counter(labels))
    # data = getCIFAR10_2label_100client()
    args.data = "CIFAR10-4label"
    args.dropout = (0.6, 0.6, 0.6)

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='a4e7793c50e34330c02de29fb678537bd57d078b')



    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 500
    for epoch in range(1, 2):
        args.epoch = epoch
        for random_seed in range(0, 1):
            args.random_seed = random_seed
            run_name = 'Four_LABEL' + "fedDYN" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
            if (args.batch_mode):
                run_name = run_name + "batch"
            wandb.init(
                project="4label-cifar10",
                name=run_name,
                config=args
            )
            set_random_seed(args.random_seed)
            model = CIFAR10_CNN(args.dropout)
            server = FedDyn(model, data, args)
            server.train()
            wandb.finish()

def run_cifar10_5label_100client():
    args = get_args()
    data = getCIFAR10_5label_each_100client()
    for user in data[0]:
        labels = [y for (x, y) in user]
        print(Counter(labels))
    # data = getCIFAR10_2label_100client()
    args.data = "CIFAR10"
    args.dropout = (0.6, 0.6, 0.6)

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='a4e7793c50e34330c02de29fb678537bd57d078b')



    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 500
    for epoch in range(1, 2):
        args.epoch = epoch
        for random_seed in range(0, 2):
            args.random_seed = random_seed
            run_name = 'Five_LABEL' + "fedDYN" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
            if (args.batch_mode):
                run_name = run_name + "batch"
            wandb.init(
                project="5label-cifar10",
                name=run_name,
                config=args
            )
            set_random_seed(args.random_seed)
            model = CIFAR10_CNN(args.dropout)
            server = FedDyn(model, data, args)
            server.train()
            wandb.finish()

def run_cifar10_7label_100client():
    args = get_args()
    data = getCIFAR10_7label_each_100client()
    for user in data[0]:
        labels = [y for (x, y) in user]
        print(Counter(labels))
    # data = getCIFAR10_2label_100client()
    args.data = "CIFAR10"

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='a4e7793c50e34330c02de29fb678537bd57d078b')



    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 300
    for epoch in range(1, 2):
        for random_seed in range(0, 2):
            args.random_seed = random_seed
            args.epoch = epoch
            run_name = 'Seven_LABEL' + "fedDYN" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
            if (args.batch_mode):
                run_name = run_name + "batch"
            wandb.init(
                project="7label-cifar10",
                name=run_name,
                config=args
            )
            set_random_seed(args.random_seed)
            model = CIFAR10_CNN(args.dropout)
            server = FedDyn(model, data, args)
            server.train()
            wandb.finish()

if __name__ == '__main__':
    # run_femnist()
    # run_mnist_2label_100client()
    # run_cifar10_4label_100client()
    # run_cifar10_3label_100client()
    # run_cifar10_4label_100client()
    # run_cifar10_5label_100client()
    # run_cifar10_7label_100client()
    # run_cifar10_2label_100client()
    # run_cifar10_2label_100client_cluster()
    run_cifar100_2label_100client()

