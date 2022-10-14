import copy
import torch
from utils.server import server
from algorithm.fedScaffold2.client import client
from algorithm.clientSelector.graidient_diversity_selector import Sampler
import random
import logging
import wandb

class FedScaffold(server):
    def __init__(self, model, data, args):
        self.model = copy.deepcopy(model)
        # a = torch.nn.utils.parameters_to_vector(self.model.parameters())
        # self.global_c = torch.zeros_like(a).cuda()
        self.set_clients(model, data, args)
        self.args = args
        if args.smart_sample:
            self.Sampler = Sampler(len(self.clients), greedy_mu=self.args.greedy_mu)

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

    def client_sampler(self, number_of_clients_per_round, round_idx):
        if self.args.smart_sample:
            print("smart_sample")
            return self.Sampler.smart_sample(number_of_clients_per_round)
        return random.sample(range(len(self.clients)), number_of_clients_per_round)

    def train(self):
        args = self.args
        self.summary(0)
        state_param_list = torch.zeros(len(self.clients) + 1, len(self.get_model_params_vector()))
        for i in range(args.comm_rounds):
            clients_in_turn = self.client_sampler(args.number_of_clients_per_round, i)
            print("client_indexes = " + str(clients_in_turn))
            logging.info("client_indexes = " + str(clients_in_turn))
            weights = []
            params = []
            cs = []
            # self.global_c = sum(c.local_c for c in self.clients)
            for idx in clients_in_turn:
                state_param_diff_curr = -state_param_list[-1] + state_param_list[-1]
                state_param_diff_curr = state_param_diff_curr.detach().cuda()
                delta_c_sum = torch.zeros(len(self.get_model_params_vector()), )
                w, param, n_minibatrch = self.clients[idx].train(self.get_model_params_vector(), state_param_diff_curr, i)
                params.append(param)
                # gradients.append(gradient) # tmp = self.get_model_params()
                # print("client ", idx, ' ',torch.norm(gradient).item(), ' ', torch.norm(c).item())
            init = self.get_model_params_vector()
            if self.args.smart_sample:
                self.Sampler.similarity_bandit([(param.cpu() - init.cpu()) for param in params], clients_in_turn)
            with torch.no_grad():
                global_learning_rate = 1
                avg_model_param = global_learning_rate*torch.mean(torch.stack(params), dim = 0)
                # init_w = self.get_model_params_vector().clone().cuda()
                # grad_global = self.weighted_average(gradients, weights)
                # delta_c_sum = self.weighted_average(cs, weights)
                # here should be a global lr
                self.set_model_params_vector(avg_model_param)
                # self.global_c = self.global_c + len(clients_in_turn)/len(self.clients)*delta_c_sum

                # self.global_c = self.global_c + torch.sum(torch.stack(cs), dim = 0)/len(self.clients)
                # self.global_c = self.global_c/2
                # self.global_c = len(clients_in_turn)/len(self.clients)*delta_c_global
                # print('### ', torch.norm(grad_global), torch.norm(self.global_c), torch.norm(delta_c_global))
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

