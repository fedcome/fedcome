import random
import numpy as np
import bisect
import torch

def dirichlet_partition(samples, num_clients, alpha):
    ret = {i: [] for i in range(num_clients)}
    random.shuffle(samples)
    prop = np.random.dirichlet(np.repeat(alpha, num_clients))
    for i in range(1, len(prop)):
        prop[i] = prop[i-1] + prop[i]
    i = 0
    for idx in range(0, len(prop)):
        pre = i
        while (i/(len(samples)) < prop[idx]):
            i += 1
        ret[idx] += samples[pre:i]
        if len(ret[idx]) == 0:
            ret[idx] += samples[pre:pre+2]
    # for i, sample in enumerate(samples):
    #     idx = bisect.bisect_left(prop, i/len(samples), 0, len(prop))
    #     ret[idx].append(sample)
    return ret

def split_by_label(data):
    ret = {}
    for sample, label in data:
        if label not in ret.keys():
            ret[label] = []
        ret[label].append((sample, label))
    return ret

def data_split(data, num_clients, alpha):
    # return a dict user2data
    label2data = split_by_label(data)
    user2data = {i: [] for i in range(num_clients)}
    for label, samples in label2data.items():
        ret = dirichlet_partition(samples, num_clients, alpha)
        for user, samples in ret.items():
            user2data[user] += ret[user]
    return list(user2data.values())

def data_to_dataloader(data, batchsize):
    # sample_tensors = torch.Tensor([s[0] for s in data])
    # target_tensors = torch.Tensor([s[1] for s in data]).T
    # dataset = torch.utils.data.TensorDataset(sample_tensors,  target_tensors)
    return torch.utils.data.DataLoader(data, batch_size = batchsize, shuffle = True)