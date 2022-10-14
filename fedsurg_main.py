
import argparse

from model.cnn import CNN_OriginalFedAvg
from model.cnn import CIFAR10_CNN
from model.cnn import MNIST_CNN
from model.cnn import HAR_MLP

from model.cnn import CIFAR100_CNN
from data.cifar100.CIFAR100 import getCIFAR100_100client
from data.cifar10.CIFAR10 import getCIFAR10_1label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_3label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_5label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_7label_each_100client
from data.har.har import getharData

from data.cifar10.CIFAR10 import getCIFAR10
from data.cifar10.CIFAR10 import getCIFAR10_2label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_2label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_4label_each_100client
from data.cifar10.CIFAR10 import getCIFAR10_2label_each_20client_for_show
from data.cifar10.CIFAR10 import number_of_classes as cifar10_number_of_classes
from data.mnist.MNIST import getMNIST_2label_100client
from data.mnist.MNIST import getMNIST_1label_100client
from data.femnist_certain.femnist import getFemnist_196clients

# from data.mnist.MNIST import getMNIST
# from data.mnist.MNIST import number_of_classes as number_of_classes
from algorithm.fedsurg.server import FedSurg
from model.cnn import Femnist_CNN
import torch
import wandb
import random
from collections import Counter


from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import numpy as np
import pickle

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

    # ret.number_of_clients = 100
    # ret.number_of_clients_per_round = 20
    # ret.alpha = 0.5

    ret.lr = 0.05
    ret.batchsize = 50
    ret.random_seed = 2


    # ret.sugery = "SIMPLE_SIMPLE_QP"
    # none, QP , init, QP_FINAL, SIMPLE_QP, SIMPLE_SIMPLE_QP
    ret.smart_sample = "none" # none, init

    ret.epoch = 1
    ret.batch_mode = False

    ret.test_frequency = 1
    ret.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ret.lr_decay = False

    # ret.model = CIFAR10_CNN((0, 0, 0))
    # ret.data = "MNIST"
    # ret.model = CNN_OriginalFedAvg([3, 32, 32], 10)
    # ret.model = MNIST_CNN()
    ret.dropout = (0.3,0.3,0.3)
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
    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    for random_seed in range(40, 45):
        args.random_seed = random_seed
        for epoch in range(1, 2):
            args.epoch = epoch
            for surgery in [ "init", "none", "SIMPLE_SIMPLE_QP"]:
                args.sugery = surgery
                if (args.sugery != "none"):
                    run_name = args.sugery + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                else:
                    run_name = "fedAvg" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                wandb.init(
                    project="fedsurgery-mnist-100client_fullparticipation",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                model = MNIST_CNN()
                server = FedSurg(model, data, args)
                result = server.train()
                file = open("./mnist_2label100client_method" + surgery + "_epoch" + str(epoch) + "_random_seed" +  str(random_seed) + ".result", 'wb')
                pickle.dump(result, file)
                file.close()


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
    for random_seed in range(101, 104):
        args.random_seed = random_seed
        args.epoch = 1
        for surgery in ["init", "none", "SIMPLE_SIMPLE_QP"]:
            args.sugery = surgery
            if (args.sugery != "none"):
                run_name = args.sugery + str(args.lr) + "lr" + str(args.epoch) + "epoch"
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
            server = FedSurg(model, data, args)
            server.train()
            wandb.finish()

def run_cifar10_2label_100client():

    args = get_args()
    data = getCIFAR10_2label_each_100client()
    # data = getCIFAR10_2label_100client()
    args.data = "CIFAR10"
    args.dropout = (0.4, 0.4, 0.4)

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')



    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 500
    args.lr_decay = True
    args.smart_sample = False
    for random_seed in range(10, 11):
        for epoch in range(1, 2):
            args.epoch = epoch
            args.random_seed = random_seed
            for surgery in ["SIMPLE_SIMPLE_QP"]:
                args.sugery = surgery
                if (args.sugery != "none"):
                    run_name = args.sugery + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                else:
                    run_name = "fedAvg" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                wandb.init(
                    project="fedSurg-CIFAR10-100-per-round-100-clients",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                model = CIFAR10_CNN()
                server = FedSurg(model, data, args)
                result = server.train()
                file = open("./cifar10_2label100client_method" + surgery + "_epoch" + str(epoch) + "_random_seed" + str(random_seed) + ".result", 'wb')
                pickle.dump(result, file)
                file.close()
                wandb.finish()

def run_cifar10_1label_100client():

        args = get_args()
        data = getCIFAR10_1label_each_100client()
        # data = getCIFAR10_2label_100client()
        model = CIFAR10_CNN((0, 0, 0))
        args.data = "CIFAR10"

        # data = getMNIST_2label_100client()
        # model = MNIST_CNN()
        wandb.login(key='')



        args.lr = 0.01
        args.number_of_clients = 100
        args.number_of_clients_per_round = 100
        args.comm_rounds = 60
        for epoch in range(1, 2):
            args.epoch = epoch
            for surgery in ["init", 'none', "SIMPLE_SIMPLE_QP"]:
            # for surgery in ["SIMPLE_SIMPLE_QP"]:
                args.sugery = surgery
                if (args.sugery != "none"):
                    run_name = "ONE_LABEL" + args.sugery + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                else:
                    run_name = 'ONE_LABEL' + "fedAvg" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                wandb.init(
                    project="fedSurg-CIFAR10-100-per-round-100-clients",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                server = FedSurg(model, data, args)
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
    args.dropout = (0.4, 0.4, 0.4)
    args.smart_sample = False

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')



    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 500
    for epoch in range(1, 2):
        args.epoch = epoch
        for surgery in ["SIMPLE_SIMPLE_QP"]:
            args.sugery = surgery
            for random_seed in range(0, 3):
                args.random_seed = random_seed
                run_name = 'Three_LABEL' + "fedCOME" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                wandb.init(
                    project="3label-cifar10",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                model = CIFAR10_CNN(args.dropout)
                server = FedSurg(model, data, args)
                server.train()
                wandb.finish()
def run_cifar10_5label_100clien():
    args = get_args()
    data = getCIFAR10_5label_each_100client()
    for user in data[0]:
        labels = [y for (x, y) in user]
        print(Counter(labels))
    # data = getCIFAR10_2label_100client()
    args.data = "CIFAR10"
    args.dropout = (0.5,0.5,0.5)

    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')



    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 300
    for epoch in range(1, 2):
        args.epoch = epoch
        for surgery in ["SIMPLE_SIMPLE_QP"]:
            args.sugery = surgery
            run_name = 'Five_LABEL' + "fedCOME" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
            if (args.batch_mode):
                run_name = run_name + "batch"
            wandb.init(
                project="5label-cifar10",
                name=run_name,
                config=args
            )
            set_random_seed(args.random_seed)
            model = CIFAR10_CNN(args.dropout)
            server = FedSurg(model, data, args)
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
    wandb.login(key='')



    args.number_of_clients = 100
    args.number_of_clients_per_round = 100
    args.comm_rounds = 300
    for epoch in range(1, 2):
        args.epoch = epoch
        for surgery in ["SIMPLE_SIMPLE_QP"]:
            args.sugery = surgery
            run_name = 'Seven_LABEL' + "fedCOM" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
            if (args.batch_mode):
                run_name = run_name + "batch"
            wandb.init(
                project="7label-cifar10",
                name=run_name,
                config=args
            )
            set_random_seed(args.random_seed)
            model = CIFAR10_CNN(args.dropout)
            server = FedSurg(model, data, args)
            server.train()
            wandb.finish()
def run_cifar100_2label_100client():

    args = get_args()
    data = getCIFAR100_100client()
    args.data = "CIFAR100"
    args.dropout = (0.7, 0.7, 0.7)
    # args.epoch = 5
    args.use_wandb = True

    if (args.use_wandb):
        wandb.login(key='')



    args.number_of_clients = 100
    args.number_of_clients_per_round = 100

    args.comm_rounds = 600
    # args.comm_rounds = 10
    args.smart_sample = False
    args.lr_decay = 0.995
    for surgery in ["SIMPLE_SIMPLE_QP"]:
        for epoch in range(5, 6):
            args.epoch = epoch
            for rand_seed in range(1):
                args.random_seed = rand_seed
                args.lr = 0.05
                args.sugery = surgery
                if (args.sugery != "none"):
                    run_name = args.sugery + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                else:
                    run_name = "fedAvg" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                if args.use_wandb:
                    wandb.init(
                        project="fedSurg-CIFAR100-100-per-round-100-clients",
                        name=run_name,
                        config=args
                    )
                set_random_seed(args.random_seed)
                model = CIFAR100_CNN(args.dropout)
                server = FedSurg(model, data, args)
                server.train()
                wandb.finish()

def test():
    args = get_args()
    set_random_seed(args.random_seed)
    # data = getCIFAR10_2label_each_100client()
    # data = getMNIST_2label_100client()
    data = getMNIST_1label_100client()
    model = MNIST_CNN()
    wandb.login(key='')
    for surgery in ["SIMPLE_SIMPLE_QP", "none"]:
        for smart_sample in [False]:
            args.sugery = surgery
            args.smart_sample = smart_sample
            if (args.sugery != "none"):
                run_name = args.sugery + str(args.lr) + "lr" +  str(args.smart_sample)
            else:
                run_name = "fedavg" + str(args.lr) + "lr" +  str(args.smart_sample)
            if (args.batch_mode):
                run_name = run_name + "batch"
            wandb.init(
                project="fedSurg",
                name=run_name,
                config=args
            )
            #
            server = FedSurg(model, data, args)
            server.train()
            wandb.finish()

def run_har():

    args = get_args()
    data = getharData()
    # data = getCIFAR10_2label_100client()
    for x, y in data[0][1]:
        print(len(x))
        break
    model = HAR_MLP()
    args.data = "HAR"

    wandb.login(key='')

    args.number_of_clients = 30
    args.number_of_clients_per_round = 30
    args.comm_rounds = 50
    for random_seed in range(4, 6):
        for epoch in range(1, 2):
            args.epoch = epoch
            for surgery in ['none', "init", "SIMPLE_SIMPLE_QP"]:
                args.sugery = surgery
                if (args.sugery != "none"):
                    run_name = args.sugery + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                else:
                    run_name = "fedAvg" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                wandb.init(
                    project="fedSurg-har-100-per-round-100-clients",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                server = FedSurg(model, data, args)
                server.train()
                wandb.finish()
def run_cifar10_4label_100client():

    args = get_args()
    args.smart_sample = False
    args.dropout = (0.5,0.5, 0.5)
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
        for surgery in ["SIMPLE_SIMPLE_QP"]:
            args.sugery = surgery
            for random_seed in range(0, 2):
                args.random_seed = random_seed
                run_name = 'Four_LABEL' + "fedCOME" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                if (args.batch_mode):
                    run_name = run_name + "batch"
                wandb.init(
                    project="4label-cifar10",
                    name=run_name,
                    config=args
                )
                set_random_seed(args.random_seed)
                model = CIFAR10_CNN(args.dropout)
                server = FedSurg(model, data, args)
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


    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')



    args.number_of_clients = 100
    # args.number_of_clients_per_round = 20
    args.comm_rounds = 500
    args.lr_decay = 1
    args.dropout = (0.3,0.3,0.3)
        # for number_of_clients_per_round in [10,20, 50]:
    # for number_of_clients_per_round, sample_num in zip([10, 20, 30, 40], [7, 10, 15, 20]):
    for number_of_clients_per_round, sample_num in zip([30], [20]):
        args.number_of_clients_per_round = number_of_clients_per_round
        args.greedy_mu = sample_num/number_of_clients_per_round + 0.01
        for epoch in range(1, 2):
            args.epoch = epoch
            for surgery in ["SIMPLE_SIMPLE_QP"]:
                args.sugery = surgery
                for smart_sample in [False]:
                    for random_seed in range(40, 42):
                        args.smart_sample = smart_sample  # none, init
                        if (args.sugery != "none"):
                            run_name = args.sugery + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                        else:
                            run_name = "fedAvg" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
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
                        server = FedSurg(model, data, args)
                        result = server.train()
                        file = open("./cifar10_2label100client_method" + surgery + "_epoch" + str(epoch) + "_random_seed" + str(random_seed) + ".result", 'wb')
                        pickle.dump(result, file)
                        file.close()
                        wandb.finish()

def verify_smart_sample():
    args = get_args()
    data = getCIFAR10_2label_each_100client()
    # data = getCIFAR10_2label_100client()
    model = CIFAR10_CNN((0, 0, 0))
    args.data = "CIFAR10"

    dataCounter = np.zeros([100, 10])
    for user_idx in range(100):
        labels = [y for (x, y) in data[0][user_idx]]
        a = Counter(labels)
        for t in range(10):
            dataCounter[user_idx][t] = a[t]
    np.save("cifar10_2label100client_cout.tb.npy", dataCounter)


    # data = getMNIST_2label_100client()
    # model = MNIST_CNN()
    wandb.login(key='')



    args.number_of_clients = 100
    # args.number_of_clients_per_round = 20
    args.comm_rounds = 20
    args.lr_decay = False
        # for number_of_clients_per_round in [10,20, 50]:
    for number_of_clients_per_round, sample_num in zip([20], [10]):
        args.number_of_clients_per_round = number_of_clients_per_round
        args.greedy_mu = sample_num/number_of_clients_per_round + 0.01
        for epoch in range(1, 2):
            args.epoch = epoch
            for surgery in ["SIMPLE_SIMPLE_QP"]:
                args.sugery = surgery
                for smart_sample in [True]:
                    for random_seed in range(40, 42):
                        args.smart_sample = smart_sample  # none, init
                        if (args.sugery != "none"):
                            run_name = args.sugery + str(args.lr) + "lr" + str(args.epoch) + "epoch"
                        else:
                            run_name = "fedAvg" + str(args.lr) + "lr" + str(args.epoch) + "epoch"
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
                        server = FedSurg(model, data, args)
                        result = server.train()
                        file = open("./cifar10_2label100client_method" + surgery + "_epoch" + str(epoch) + "_random_seed" + str(random_seed) + ".result", 'wb')
                        pickle.dump(result, file)
                        file.close()
                        wandb.finish()
if __name__ == '__main__':
    # run_mnist_2label_100client()
    run_cifar10_2label_100client_cluster()
    # run_cifar10_4label_100client()
    # run_cifar10_2label_100client()
    # run_cifar10_1label_100client()
    # run_cifar10_3label_100client()
    # run_cifar10_5label_100client()
    # run_cifar10_4label_100client()
    # run_cifar10_7label_100client()
    # run_cifar100_2label_100client()
    # verify_smart_sample()
    # run_femnist()
    # run_har()
    # test()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
