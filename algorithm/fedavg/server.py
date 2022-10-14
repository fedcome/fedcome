import copy
import torch
from utils.server import server
from algorithm.fedavg.client import client
import random
import logging
import wandb

class FedAvg(server):
    def __init__(self, model, data, args):
        self.model = copy.deepcopy(model)
        self.set_clients(model, data, args)
        self.args = args

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def set_clients(self, model, data, args):
        self.clients = [client(model, local_data, args) for local_data in data]
        return

    def client_sampler(self, number_of_clients_per_round):
        return random.sample(range(len(self.clients)), number_of_clients_per_round)

    def train(self):
        args = self.args
        self.summary(0)
        for i in range(args.comm_rounds):
            clients_in_turn = self.client_sampler(args.number_of_clients_per_round)
            logging.info("client_indexes = " + str(clients_in_turn))
            weights = []
            updates = []
            print("test for each")
            for idx in clients_in_turn:
                w, param = self.clients[idx].train(self.get_model_params())
                weights.append(w)
                updates.append(param)
                tmp = self.get_model_params()
                self.set_model_params(updates[-1])
                self.summary(i, use_wandb = False)
                self.set_model_params(tmp)
            print("####test for each END")
            w_global = self.aggregate(updates, weights)
            self.set_model_params(w_global)
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

