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
number_of_classes = 100
#
def getCIFAR100_100client(random_seed = 1):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # file = cur_dir + "/" + str(num_clients) + "_" +  str(alpha) + "_" + str(random_seed) + ".pkl"
    # if os.path.exists(file):
    #     rf = open(file, "rb")
    #     user2data = pickle.load(rf)
    #     rf.close()
    #     return user2data


    np.random.seed(random_seed)
    random.seed(random_seed)

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ]
    )
    training_data = torchvision.datasets.CIFAR100(
        root=cur_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_data = torchvision.datasets.CIFAR100(
        root=cur_dir,
        train=False,
        download=True,
        transform=transform
    )
    superclass = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                  ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                  ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                  ['bottle', 'bowl', 'can', 'cup', 'plate'],
                  ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                  ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                  ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                  ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                  ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                  ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                  ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                  ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                  ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                  ['crab', 'lobster', 'snail', 'spider', 'worm'],
                  ['baby', 'boy', 'girl', 'man', 'woman'],
                  ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                  ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                  ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                  ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                  ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]

    idx2data = [[] for i in range(100)]
    for x, y in training_data:
        idx2data[y].append((x, y))
    superclass2data = []
    for vct in superclass:
        cur_class = []
        for name in vct:
            cur_class.append(idx2data[training_data.class_to_idx[name]])
        superclass2data.append(cur_class)
    user2data_train = []
    for classdata in superclass2data:
        whole_data = []
        for v in classdata:
            whole_data += v
        whole_data += whole_data
        user2data_train += [whole_data[250 + i*500: 250 + i*500 + 500] for i in range(5)]

    whole_data = [(x, y) for (x, y) in test_data]
    user2data_test = [whole_data[i*100:i*100 + 100] for i in range(100)]
    return user2data_train, user2data_test




def main():

    getCIFAR100_100client(10)


if __name__ == '__main__':
    main()