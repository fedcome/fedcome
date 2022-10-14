# import sys
# sys.path.append("...")
#
from utils.data_process import data_split
#
#
import os
import torchvision
#
import random
import numpy as np
import pickle
import heapq
#
#
number_of_classes = 10
#
def getCIFAR10(num_clients, alpha, random_seed = 1):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    file = cur_dir + "/" + str(num_clients) + "_" +  str(alpha) + "_" + str(random_seed) + ".pkl"
    if os.path.exists(file):
        rf = open(file, "rb")
        user2data = pickle.load(rf)
        rf.close()
        return user2data


    np.random.seed(random_seed)
    random.seed(random_seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    training_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=False,
        download=True,
        transform=transform
    )
    whole_data = [(x, y) for x, y in training_data]
    whole_data += [(x, y) for x, y in test_data]
    user2data = data_split(whole_data, num_clients, alpha)

    wf = open(file, "wb")
    pickle.dump(user2data, wf)
    wf.close()
    return user2data

def getCIFAR10_1label_each_10client(random_seed = 1):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # file = cur_dir + "/" + "2label_each_10client_" + str(10) + "_" + str(random_seed) + ".pkl"
    # if os.path.exists(file):
    #     rf = open(file, "rb")
    #     user2data_train, user2data_test = pickle.load(rf)
    #     rf.close()
    #     return user2data_train, user2data_test

    np.random.seed(random_seed)
    random.seed(random_seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    training_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=False,
        download=True,
        transform=transform
    )
    whole_data = [(x, y) for x, y in training_data]
    # whole_data += [(x, y) for x, y in test_data]

    whole_data.sort(key = lambda data: data[1])
    whole_data += whole_data
    user2data_train = [whole_data[i*5000+2500: i*5000 + 2500+ 5000] for i in range(10)]

    whole_data = [(x, y) for x, y in test_data]
    whole_data += whole_data
    user2data_test = [whole_data[i*1000: (i+1)*1000] for i in range(10)]


    # user2dataloader = {}
    # for user, samples in user2data.items():
    #     user2dataloader[user] = samples_to_dataloader(samples, batchsize)

    # wf = open(file, "wb")
    # pickle.dump((user2data_train, user2data_test), wf)
    # wf.close()
    return user2data_train, user2data_test

def getCIFAR10_1label_each_100client(random_seed = 1):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # file = cur_dir + "/" + "2label_each_10client_" + str(10) + "_" + str(random_seed) + ".pkl"
    # if os.path.exists(file):
    #     rf = open(file, "rb")
    #     user2data_train, user2data_test = pickle.load(rf)
    #     rf.close()
    #     return user2data_train, user2data_test

    np.random.seed(random_seed)
    random.seed(random_seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    training_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=False,
        download=True,
        transform=transform
    )
    whole_data = [(x, y) for x, y in training_data]
    # whole_data += [(x, y) for x, y in test_data]

    whole_data.sort(key = lambda data: data[1])
    # whole_data += whole_data
    user2data_train = [whole_data[i*500: i*500 + 500] for i in range(100)]

    whole_data = [(x, y) for x, y in test_data]
    # whole_data += whole_data
    user2data_test = [whole_data[i*100: (i+1)*100] for i in range(100)]


    # user2dataloader = {}
    # for user, samples in user2data.items():
    #     user2dataloader[user] = samples_to_dataloader(samples, batchsize)

    # wf = open(file, "wb")
    # pickle.dump((user2data_train, user2data_test), wf)
    # wf.close()
    return user2data_train, user2data_test

def getCIFAR10_2label_each_100client(random_seed = 1):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # file = cur_dir + "/" + "2label_each_100client_" + str(10) + "_" + str(random_seed) + ".pkl"
    # if os.path.exists(file):
    #     rf = open(file, "rb")
    #     user2data_train, test_data = pickle.load(rf)
    #     rf.close()
    #     return user2data_train, test_data

    np.random.seed(random_seed)
    random.seed(random_seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    training_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=False,
        download=True,
        transform=transform
    )
    train_data = [(x, y) for x, y in training_data]
    # whole_data += [(x, y) for x, y in test_data]
    random.shuffle(train_data)
    # 50000/100 = 500, 250 per class

    label2idx = {i: [] for i in range(10)}

    for i, (x, y) in enumerate(train_data):
        label2idx[y].append(i)
    queue = []
    for i in range(10):
        random.shuffle(label2idx[i])
        queue.append((-len(label2idx[i]), label2idx[i]))
    heapq.heapify(queue)
    user2data_train = []
    while len(queue) >= 2:
        ret = []
        for i in range(2):
            cnt, lst = heapq.heappop(queue)
            ret += lst[0:250]
            lst = lst[250:]
            cnt += 250
            if len(lst) < 250:
                ret += lst
                lst = []
                cnt = 0
            if cnt != 0:
                heapq.heappush(queue, (cnt, lst))
        user2data_train.append([train_data[i] for i in ret])

    test_data = [(x, y) for x, y in test_data]
    user2data_test = [test_data[i*100:i*100 + 100] for i in range(100)]


    # whole_data.sort(key = lambda data: data[1])
    # whole_data += whole_data
    # user2data = [whole_data[i*6000+3000: i*6000 +3000 + 6000] for i in range(10)]



    # user2dataloader = {}
    # for user, samples in user2data.items():
    #     user2dataloader[user] = samples_to_dataloader(samples, batchsize)

    # wf = open(file, "wb")
    # pickle.dump((user2data_train, user2data_test), wf)
    # wf.close()
    return user2data_train, user2data_test

def getCIFAR10_2label_each_20client_for_show(random_seed = 1):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # file = cur_dir + "/" + "2label_each_100client_" + str(10) + "_" + str(random_seed) + ".pkl"
    # if os.path.exists(file):
    #     rf = open(file, "rb")
    #     user2data_train, test_data = pickle.load(rf)
    #     rf.close()
    #     return user2data_train, test_data

    np.random.seed(random_seed)
    random.seed(random_seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    training_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=False,
        download=True,
        transform=transform
    )
    train_data = [(x, y) for x, y in training_data]
    # whole_data += [(x, y) for x, y in test_data]
    random.shuffle(train_data)
    # 50000/100 = 500, 250 per class

    label2idx = {i: [] for i in range(10)}

    for i, (x, y) in enumerate(train_data):
        label2idx[y].append(i)

    sz = len(label2idx[0])//4
    user2data_train = []
    for i in range(10):
        x, y = i, (i+1)%10
        for j in range(2):
            data_i = label2idx[x][:sz] + label2idx[y][:sz]
            label2idx[x] = label2idx[x][sz:]
            label2idx[y] = label2idx[y][sz:]
            user2data_train.append([train_data[idx] for idx in data_i])
            print(len(label2idx[x]), len(label2idx[y]))

    test_data = [(x, y) for x, y in test_data]
    user2data_test = [test_data[i*500:i*500 + 500] for i in range(20)]


    return user2data_train, user2data_test
def getCIFAR10_5label_each_100client(random_seed = 1):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # file = cur_dir + "/" + "2label_each_100client_" + str(10) + "_" + str(random_seed) + ".pkl"
    # if os.path.exists(file):
    #     rf = open(file, "rb")
    #     user2data_train, test_data = pickle.load(rf)
    #     rf.close()
    #     return user2data_train, test_data

    np.random.seed(random_seed)
    random.seed(random_seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    training_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=False,
        download=True,
        transform=transform
    )
    train_data = [(x, y) for x, y in training_data]
    # whole_data += [(x, y) for x, y in test_data]
    random.shuffle(train_data)
    # 50000/100 = 500, 250 per class

    label2idx = {i: [] for i in range(10)}

    for i, (x, y) in enumerate(train_data):
        label2idx[y].append(i)
    queue = []
    for i in range(10):
        random.shuffle(label2idx[i])
        queue.append((-len(label2idx[i]), label2idx[i]))
    heapq.heapify(queue)
    user2data_train = []
    while len(queue) >= 5:
        ret = []
        for i in range(5):
            cnt, lst = heapq.heappop(queue)
            ret += lst[0:100]
            lst = lst[100:]
            cnt += 100
            if len(lst) < 100:
                ret += lst
                lst = []
                cnt = 0
            if cnt != 0:
                heapq.heappush(queue, (cnt, lst))
        user2data_train.append([train_data[i] for i in ret])

    test_data = [(x, y) for x, y in test_data]
    user2data_test = [test_data[i*100:i*100 + 100] for i in range(100)]


    # whole_data.sort(key = lambda data: data[1])
    # whole_data += whole_data
    # user2data = [whole_data[i*6000+3000: i*6000 +3000 + 6000] for i in range(10)]



    # user2dataloader = {}
    # for user, samples in user2data.items():
    #     user2dataloader[user] = samples_to_dataloader(samples, batchsize)

    # wf = open(file, "wb")
    # pickle.dump((user2data_train, user2data_test), wf)
    # wf.close()
    return user2data_train, user2data_test

def getCIFAR10_2label_each_20client_for_show(random_seed = 1):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # file = cur_dir + "/" + "2label_each_100client_" + str(10) + "_" + str(random_seed) + ".pkl"
    # if os.path.exists(file):
    #     rf = open(file, "rb")
    #     user2data_train, test_data = pickle.load(rf)
    #     rf.close()
    #     return user2data_train, test_data

    np.random.seed(random_seed)
    random.seed(random_seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    training_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=False,
        download=True,
        transform=transform
    )
    train_data = [(x, y) for x, y in training_data]
    # whole_data += [(x, y) for x, y in test_data]
    random.shuffle(train_data)
    # 50000/100 = 500, 250 per class

    label2idx = {i: [] for i in range(10)}

    for i, (x, y) in enumerate(train_data):
        label2idx[y].append(i)

    sz = len(label2idx[0])//4
    user2data_train = []
    for i in range(10):
        x, y = i, (i+1)%10
        for j in range(2):
            data_i = label2idx[x][:sz] + label2idx[y][:sz]
            label2idx[x] = label2idx[x][sz:]
            label2idx[y] = label2idx[y][sz:]
            user2data_train.append([train_data[idx] for idx in data_i])
            print(len(label2idx[x]), len(label2idx[y]))

    test_data = [(x, y) for x, y in test_data]
    user2data_test = [test_data[i*500:i*500 + 500] for i in range(20)]


    return user2data_train, user2data_test
def getCIFAR10_4label_each_100client(random_seed = 1):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # file = cur_dir + "/" + "2label_each_100client_" + str(10) + "_" + str(random_seed) + ".pkl"
    # if os.path.exists(file):
    #     rf = open(file, "rb")
    #     user2data_train, test_data = pickle.load(rf)
    #     rf.close()
    #     return user2data_train, test_data

    np.random.seed(random_seed)
    random.seed(random_seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    training_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=False,
        download=True,
        transform=transform
    )
    train_data = [(x, y) for x, y in training_data]
    # whole_data += [(x, y) for x, y in test_data]
    random.shuffle(train_data)
    # 50000/100 = 500, 250 per class

    label2idx = {i: [] for i in range(10)}

    for i, (x, y) in enumerate(train_data):
        label2idx[y].append(i)
    queue = []
    for i in range(10):
        random.shuffle(label2idx[i])
        queue.append((-len(label2idx[i]), label2idx[i]))
    heapq.heapify(queue)
    user2data_train = []
    while len(queue) >= 4:
        ret = []
        for i in range(4):
            cnt, lst = heapq.heappop(queue)
            ret += lst[0:125]
            lst = lst[125:]
            cnt += 125
            if len(lst) < 125:
                ret += lst
                lst = []
                cnt = 0
            if cnt != 0:
                heapq.heappush(queue, (cnt, lst))
        user2data_train.append([train_data[i] for i in ret])

    test_data = [(x, y) for x, y in test_data]
    user2data_test = [test_data[i*100:i*100 + 100] for i in range(100)]


    # whole_data.sort(key = lambda data: data[1])
    # whole_data += whole_data
    # user2data = [whole_data[i*6000+3000: i*6000 +3000 + 6000] for i in range(10)]



    # user2dataloader = {}
    # for user, samples in user2data.items():
    #     user2dataloader[user] = samples_to_dataloader(samples, batchsize)

    # wf = open(file, "wb")
    # pickle.dump((user2data_train, user2data_test), wf)
    # wf.close()
    return user2data_train, user2data_test


def getCIFAR10_3label_each_100client(random_seed = 1):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # file = cur_dir + "/" + "2label_each_100client_" + str(10) + "_" + str(random_seed) + ".pkl"
    # if os.path.exists(file):
    #     rf = open(file, "rb")
    #     user2data_train, test_data = pickle.load(rf)
    #     rf.close()
    #     return user2data_train, test_data

    np.random.seed(random_seed)
    random.seed(random_seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    training_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=False,
        download=True,
        transform=transform
    )
    train_data = [(x, y) for x, y in training_data]
    # whole_data += [(x, y) for x, y in test_data]
    random.shuffle(train_data)
    # 50000/100 = 500, 250 per class

    label2idx = {i: [] for i in range(10)}

    for i, (x, y) in enumerate(train_data):
        label2idx[y].append(i)
    queue = []
    for i in range(10):
        random.shuffle(label2idx[i])
        queue.append((-len(label2idx[i]), label2idx[i]))
    heapq.heapify(queue)
    user2data_train = []
    while len(queue) >= 3:
        ret = []
        for i in range(3):
            cnt, lst = heapq.heappop(queue)
            ret += lst[0:166]
            lst = lst[166:]
            cnt += 166
            if len(lst) < 166:
                ret += lst
                lst = []
                cnt = 0
            if cnt != 0:
                heapq.heappush(queue, (cnt, lst))
        user2data_train.append([train_data[i] for i in ret])

    test_data = [(x, y) for x, y in test_data]
    user2data_test = [test_data[i*100:i*100 + 100] for i in range(100)]


    # whole_data.sort(key = lambda data: data[1])
    # whole_data += whole_data
    # user2data = [whole_data[i*6000+3000: i*6000 +3000 + 6000] for i in range(10)]



    # user2dataloader = {}
    # for user, samples in user2data.items():
    #     user2dataloader[user] = samples_to_dataloader(samples, batchsize)

    # wf = open(file, "wb")
    # pickle.dump((user2data_train, user2data_test), wf)
    # wf.close()
    return user2data_train, user2data_test

def getCIFAR10_7label_each_100client(random_seed = 1):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # file = cur_dir + "/" + "2label_each_100client_" + str(10) + "_" + str(random_seed) + ".pkl"
    # if os.path.exists(file):
    #     rf = open(file, "rb")
    #     user2data_train, test_data = pickle.load(rf)
    #     rf.close()
    #     return user2data_train, test_data

    np.random.seed(random_seed)
    random.seed(random_seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )
    training_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR10(
        root=cur_dir,
        train=False,
        download=True,
        transform=transform
    )
    train_data = [(x, y) for x, y in training_data]
    # whole_data += [(x, y) for x, y in test_data]
    random.shuffle(train_data)
    # 50000/100 = 500, 250 per class

    label2idx = {i: [] for i in range(10)}

    for i, (x, y) in enumerate(train_data):
        label2idx[y].append(i)
    queue = []
    for i in range(10):
        random.shuffle(label2idx[i])
        queue.append((-len(label2idx[i]), label2idx[i]))
    heapq.heapify(queue)
    user2data_train = []
    k = 7
    sz = 500//7
    while len(queue) >= k:
        ret = []
        for i in range(k):
            cnt, lst = heapq.heappop(queue)
            ret += lst[0:sz]
            lst = lst[sz:]
            cnt += sz
            if len(lst) < sz:
                ret += lst
                lst = []
                cnt = 0
            if cnt != 0:
                heapq.heappush(queue, (cnt, lst))
        user2data_train.append([train_data[i] for i in ret])

    test_data = [(x, y) for x, y in test_data]
    user2data_test = [test_data[i*100:i*100 + 100] for i in range(100)]


    # whole_data.sort(key = lambda data: data[1])
    # whole_data += whole_data
    # user2data = [whole_data[i*6000+3000: i*6000 +3000 + 6000] for i in range(10)]



    # user2dataloader = {}
    # for user, samples in user2data.items():
    #     user2dataloader[user] = samples_to_dataloader(samples, batchsize)

    # wf = open(file, "wb")
    # pickle.dump((user2data_train, user2data_test), wf)
    # wf.close()
    return user2data_train, user2data_test
def main():
    # user2data_train, test = getCIFAR10_2label_each_100client()
    user2data_train, test = getCIFAR10_2label_each_100client()


    getCIFAR10(100, 20, 10)


if __name__ == '__main__':
    main()