import copy
import torch
from utils.server import server
# from algorithm.fedsurg.client import client
from algorithm.fedProx.client import client
import random
import logging
import wandb
from algorithm.clientSelector.graidient_diversity_selector import Sampler
import sklearn
import scipy
from sklearn.decomposition import PCA
import pickle

from sklearn.svm import LinearSVC
import numpy as np
import cvxopt
import math
import matplotlib.pyplot as plt

class FedProx(server):
    def __init__(self, model, data, args):
        self.model = copy.deepcopy(model)
        self.set_clients(model, data, args)
        self.args = copy.deepcopy(args)
        # for smart sample
        if self.args.smart_sample:
            n = len(self.clients)
            self.sampler = Sampler(number_of_clients = n, greedy_mu = self.args.greedy_mu)

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_model_params_vector(self):
        return torch.nn.utils.parameters_to_vector(self.model.parameters())

    def set_model_params_vector(self, param):
        return torch.nn.utils.vector_to_parameters(param, self.model.parameters())

    def set_clients(self, model, data, args):
        self.clients = [client(model, local_data, args) for local_data in zip(data[0], data[1])]
        return

    def client_sampler(self, number_of_clients_per_round, round_idx):
        if self.args.smart_sample:
            print("smart sample")
            return self.sampler.smart_sample(number_of_clients_per_round)
        return random.sample(range(len(self.clients)), number_of_clients_per_round)

    def train(self):
        args = self.args
        self.summary(0)
        self.sampled_clients_before = set()
        result = []
        for i in range(args.comm_rounds):
            clients_in_turn = self.client_sampler(args.number_of_clients_per_round, i)
            wandb.log({"statistic data": len(self.sampled_clients_before.intersection(set(clients_in_turn))), "round": i + 1})
            self.sampled_clients_before = set(clients_in_turn)
            print("client_indexes = " + str(sorted(clients_in_turn)))
            logging.info("client_indexes = " + str(clients_in_turn))
            weights = []
            gradients = []
            # print("test for each")
            for idx in clients_in_turn:
                w, gradient = self.clients[idx].train(self.get_model_params(), i+1)
                weights.append(w)
                gradients.append(gradient) # tmp = self.get_model_params()
            if args.smart_sample:
                self.sampler.similarity_bandit(gradients, clients_in_turn)
            init_w = self.get_model_params_vector().clone().cuda()
            grad_global = self.weighted_average(gradients, weights)

            self.set_model_params_vector(init_w+grad_global)
            if (i % args.test_frequency == 0):
                tmp = self.summary(i+1)
                result.append((i+1, tmp))
        self.report_clients()
        return result


    def report_clients(self):
        for i, client in enumerate(self.clients):
            summary, weight = client.report_train(self.get_model_params())

            wandb.log({"Final Train/Acc": summary["train_precision"], "clientid": i})
            wandb.log({"Final Train/Loss": summary["train_loss"], "clientid": i})

            summary, weight = client.report_test(self.get_model_params())

            wandb.log({"Final Test/Acc": summary["test_precision"], "clientid": i})
            wandb.log({"Final Test/Loss": summary["test_loss"], "clientid": i})

    def summary_test(self):
        summaries, weights = [], []
        for client in self.clients:
            summary, weight = client.report_test(self.get_model_params())
            summaries.append(summary)
            weights.append(weight)
        test_summary = self.aggregate(summaries, weights)
        return test_summary["test_loss"], test_summary["test_precision"]

    def summary_train(self):
        summaries, weights = [], []
        for client in self.clients:
            summary, weight = client.report_train(self.get_model_params())
            summaries.append(summary)
            weights.append(weight)
        train_summary = self.aggregate(summaries, weights)
        return train_summary["train_loss"], train_summary["train_precision"]

    def summary(self, round_idx, use_wandb = True):
        test_loss, test_precision = self.summary_test()
        train_loss, train_precision = self.summary_train()

        if use_wandb:
            wandb.log({"Train/Acc": train_precision, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})

            wandb.log({"Test/Acc": test_precision, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})

        print("rid, test_loss, test_precision: ", round_idx, test_loss, " ", test_precision)
        print("rid, traing_loss, train_precision: ", round_idx, train_loss, " ", train_precision)
        return (test_loss, test_precision, train_loss, train_precision)


    def encode(self, updates, args=None):
        print("hello encode")

    def decode(self, zip_updates, args=None):
        print("hello decode")

    def weighted_average(self, nums, weights):
        return sum([num*w for num, w in zip(nums, weights)])/sum(weights)

    def aggregate(self, updates, weights):
        ret = dict.fromkeys(updates[0].keys())
        for k in ret.keys():
            ret[k] = self.weighted_average([x[k] for x in updates], weights)
        return ret

