import copy
import torch
from utils.server import server
from algorithm.fedDC.client import client
from algorithm.clientSelector.graidient_diversity_selector import Sampler
import numpy as np
import random
import logging
import wandb

class FedDC(server):
    def __init__(self, model, data, args):
        self.model = copy.deepcopy(model)
        # a = torch.nn.utils.parameters_to_vector(self.model.parameters())
        # self.global_c = torch.zeros_like(a).cuda()
        self.set_clients(model, data, args)
        self.args = args
        if self.args.smart_sample:
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
        if self.args.smart_sample and round_idx > 10:
            print("smart_sample")
            return self.Sampler.smart_sample(number_of_clients_per_round)
        return random.sample(range(len(self.clients)), number_of_clients_per_round)

    def train(self):
        args = self.args
        self.summary(0)
        n_par = len(self.get_model_params_vector())
        n_clnt = len(self.clients)
        lr = self.args.lr


        # state_gradient_diff = np.zeros((len(self.clients) + 1, npar)).astype('float32')
        # parameter_drifts = np.zeros((len(self.clients), npar)).astype('float32')

        parameter_drifts = np.zeros((n_clnt, n_par)).astype('float32')
        init_par_list= self.get_model_params_vector().cpu().numpy()
        clnt_params_list  = np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
        # clnt_models = list(range(n_clnt))
        # saved_itr = -1

        ###
        state_gradient_diffs = np.zeros((n_clnt+1, n_par)).astype('float32') #including cloud state
    

        for i in range(args.comm_rounds):
            delta_g_sum = np.zeros(n_par)
            clients_in_turn = self.client_sampler(args.number_of_clients_per_round, i)
            print("client_indexes = " + str(clients_in_turn))
            logging.info("client_indexes = " + str(clients_in_turn))
            weights = []
            params = []
            cs = []
            init = self.get_model_params_vector()
            # self.global_c = sum(c.local_c for c in self.clients)
            device = self.args.device

            delta_grad = []
            for idx in clients_in_turn:

                local_update_last = state_gradient_diffs[idx]
                # TO DO weight list
                global_update_last = state_gradient_diffs[-1]
                alpha = self.args.alpha_coef
                # end TO DO
                hist_i = torch.tensor(parameter_drifts[idx], dtype=torch.float32, device=device)
                w, curr_model_par, n_minibatch = self.clients[idx].train(self.get_model_params_vector(), 
                                                    global_update_last, alpha, local_update_last, hist_i, i)
                params.append(curr_model_par)
                delta_param_curr = (curr_model_par - init.cuda()).detach().cpu().numpy()
                delta_grad.append(delta_param_curr)

                parameter_drifts[idx] += delta_param_curr
                beta = 1/n_minibatch/lr

                state_g = local_update_last - global_update_last + beta*(-delta_param_curr)
                # TO DO insert weight_list
                delta_g_cur = (state_g - state_gradient_diffs[idx])
                delta_g_sum += delta_g_cur
                state_gradient_diffs[idx] = state_g
                clnt_params_list[idx] = curr_model_par.cpu().detach().numpy()
                # gradients.append(gradient) # tmp = self.get_model_curr_models()
                # print("client ", idx, ' ',torch.norm(gradient).item(), ' ', torch.norm(c).item())
            if self.args.smart_sample:
                self.Sampler.similarity_bandit([torch.tensor(x) for x in delta_grad], clients_in_turn)
            avg_mdl_param_sel = torch.mean(torch.stack(params), dim = 0).detach()
            delta_g_cur = 1/n_clnt*delta_g_sum
            state_gradient_diffs[-1] += delta_g_cur
            cld_mdl_param = avg_mdl_param_sel.cpu().numpy() + np.mean(parameter_drifts, axis = 0)
            # cld_mdl_curr_model = avg_model_curr_model.detach().cpu().numpy() + np.mean(curr_modeleter_drifts, axis = 0)
            self.set_model_params_vector(torch.tensor(cld_mdl_param))
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

