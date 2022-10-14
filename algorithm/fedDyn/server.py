import copy
import torch
from utils.server import server
from algorithm.fedDyn.client import client
from algorithm.clientSelector.graidient_diversity_selector import Sampler
import random
import logging
import wandb
import sklearn
import scipy
from sklearn.decomposition import PCA

from sklearn.svm import LinearSVC
import numpy as np
import math

class FedDyn(server):
    def __init__(self, model, data, args):
        self.model = copy.deepcopy(model)
        # self.params = copy.deepcopy(model.flat_w)
        # self.h = torch.zeros_like(self.model.flat_w).cpu()
        self.set_clients(model, data, args)
        if args.smart_sample:
            self.Sampler = Sampler(len(self.clients), greedy_mu=args.greedy_mu)
        self.args = args

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def get_model_params_vector(self):
        return copy.deepcopy(torch.nn.utils.parameters_to_vector(self.model.parameters()).detach())

    def set_model_params_vector(self, param):
        return torch.nn.utils.vector_to_parameters(param, self.model.parameters())

    def set_clients(self, model, data, args):
        self.clients = [client(model, local_data, args) for local_data in zip(data[0], data[1])]
        return

    def client_sampler(self, number_of_clients_per_round):
        if self.args.smart_sample:
            print("smart_sample")
            return self.Sampler.smart_sample(number_of_clients_per_round)
        return random.sample(range(len(self.clients)), number_of_clients_per_round)

    def train(self):
        args = self.args
        self.summary(0)
        n_clnt = self.args.number_of_clients
        n_par = len(self.get_model_params_vector())
        n = self.args.number_of_clients_per_round
        # alpha = self.args.alpha
        device = self.args.device
        # self.average_model = self.get_model_params()
        # self.cld_mdl_params = self.get_model_params()
        hist_params_diffs = np.zeros((n_clnt, n_par)).astype(np.float32)

        for i in range(args.comm_rounds):
            clients_in_turn = self.client_sampler(args.number_of_clients_per_round)
            print("client_indexes = " + str(clients_in_turn))
            logging.info("client_indexes = " + str(clients_in_turn))
            weights = []
            params = []
            # print("test for each")

            init = self.get_model_params_vector()
            for idx in clients_in_turn:
                # TODOweights list
                alpha_coef_adpt = self.args.alpha_coef
                hist_params_diffs_curr = torch.tensor(hist_params_diffs[idx], dtype = torch.float32, device = device)
                w, curr_model_par = self.clients[idx].train(self.get_model_params_vector(), alpha_coef_adpt, hist_params_diffs_curr, i)
                weights.append(w)
                params.append(curr_model_par.cpu()) # tmp = self.get_model_params()
                hist_params_diffs[idx] += curr_model_par.cpu().numpy() - init.cpu().numpy()

            # self.cld_mdl_params = self.average_model + sum(c.prev_grads for c in self.clients)/args.number_of_clients
            # self.set_model_params(self.cld_mdl_params)
            if self.args.smart_sample:
                print('similarity_bandit')
                self.Sampler.similarity_bandit([(param.cpu() - init.cpu()) for param in params], clients_in_turn)
            with torch.no_grad():
                avg_mdl_param_sel = torch.mean(torch.stack(params), dim = 0).detach()
                cld_mdl_param = avg_mdl_param_sel.cpu().numpy() + np.mean(hist_params_diffs, axis = 0)

                self.set_model_params_vector(torch.tensor(cld_mdl_param))
                # self.h = self.h - 1/m*sum(gradients)
                # self.average_model = sum([self.clients[idx].get_model_params_vector() for idx in clients_in_turn])/n
            if (i % args.test_frequency == 0):
                self.summary(i+1)

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

