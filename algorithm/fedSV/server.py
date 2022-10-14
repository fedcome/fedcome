import copy
import torch
from utils.server import server
from algorithm.fedSV.client import client
import random
import logging
import wandb
from algorithm.clientSelector.graidient_diversity_selector import Sampler
import sklearn
import scipy
from sklearn.decomposition import PCA

from sklearn.svm import LinearSVC
import numpy as np
import math

class FedSV(server):
    def __init__(self, model, data, args):
        self.model = copy.deepcopy(model)
        self.set_clients(model, data, args)
        self.client_gradients_backup = []
        self.args = args
        if args.smart_sample:
            self.sampler = Sampler(args.number_of_clients, greedy_mu=self.args.greedy_mu)


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
            print("smart_sample")
            return self.sampler.smart_sample(number_of_clients_per_round, round_idx)
        return random.sample(range(len(self.clients)), number_of_clients_per_round)

    def train(self):
        args = self.args
        self.summary(0)
        for i in range(args.comm_rounds):
            clients_in_turn = self.client_sampler(args.number_of_clients_per_round, i+1)
            print("client_indexes = " + str(clients_in_turn))
            logging.info("client_indexes = " + str(clients_in_turn))
            weights = []
            gradients = []
            losses = []
            # print("test for each")

            for idx in clients_in_turn:
                w, gradient, loss = self.clients[idx].train(self.get_model_params(), i)
                weights.append(w)
                gradients.append(gradient) # tmp = self.get_model_params()
                losses.append(loss)

            if args.smart_sample:
                self.sampler.similarity_bandit(gradients, clients_in_turn)
            with torch.no_grad():
                init_w = self.get_model_params_vector().clone().cpu()
                # self.client_gradients_backup.append(i, gradients)
                # self.client_gradients_backup = self.client_gradients_backup[-args.tau:]
                fake_grad_global = self.weighted_average(gradients, weights)

                gradients = self.internal_gradients_surgery(gradients, losses)
                grad_global = self.weighted_average(gradients, weights)
                if args.external_surgery:
                    grad_global = self.external_gradients_surgery(grad_global, i)
                print('norm init', torch.norm(fake_grad_global))
                print('after init', torch.norm(grad_global))
                grad_global = grad_global / torch.norm(grad_global) * torch.norm(fake_grad_global)
                self.set_model_params_vector(init_w + grad_global.cpu())
            if (i % args.test_frequency == 0):
                self.summary(i+1)

    def internal_gradients_surgery(self, grads, losses):
        proj_grads = [x.clone() for x in grads]
        tmp = sorted(list(zip(losses, losses)), key=lambda x: x[0])
        order = [_ for _ in range(len(proj_grads))]

        # sort client gradients according to their losses in ascending orders
        tmp = sorted(list(zip(losses, order)), key=lambda x: x[0])
        order = [x[1] for x in tmp]
        keep_original = []
        if self.args.alpha > 0:
            keep_original = order[math.ceil((len(order) - 1) * (1 - self.args.alpha)):]
        print("keep original:", keep_original)
        with torch.no_grad():
            minus_cnt = 0
            for i in range(len(grads)):
                if i in keep_original:
                    continue
                for j in range(len(grads)):
                    if (i == j): continue
                    t = torch.dot(proj_grads[i], grads[j])
                    # print ("dot product", t.item(), (t/torch.norm(grads[i])/torch.norm(grads[j])).item())
                    if t < 0:
                        minus_cnt = minus_cnt + 1
                        proj_grads[i] = proj_grads[i] - grads[j]*t / (torch.norm(grads[j]) ** 2)
            print("minus cnt = ", minus_cnt)
            return proj_grads

    def external_gradients_surgery(self, grad_global, round_idx):
        with torch.no_grad():
            grad_global = grad_global.cuda()
            for curr_tau in range(self.args.tau-1, -1, -1):
                g_con = torch.zeros_like(grad_global).cuda()
                for c in self.clients:
                    if c.last_round is None:
                        continue
                    if c.last_round < round_idx - self.args.tau:
                        continue
                    if c.last_round == round_idx - curr_tau:
                        grad = c.last_grad.cuda()
                        t = torch.dot(grad, grad_global)
                        if (t < 0):
                            g_con += grad
                t = torch.dot(g_con, grad_global)
                print('tau, dot = ', curr_tau, t)
                if t < 0:
                    grad_global = grad_global - t / (torch.norm(g_con) ** 2) * g_con

        return grad_global



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

