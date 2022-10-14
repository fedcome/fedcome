import copy
import torch
from utils.server import server
from algorithm.fedsurg.client import client
import random
import logging
import wandb
import sklearn
import scipy
from sklearn.decomposition import PCA
from algorithm.clientSelector.graidient_diversity_selector import Sampler
import pickle

from sklearn.svm import LinearSVC
import numpy as np
import cvxopt
import math
import matplotlib.pyplot as plt

class FedSurg(server):
    def __init__(self, model, data, args):
        self.model = copy.deepcopy(model)
        self.set_clients(model, data, args)
        self.args = copy.deepcopy(args)
        ## for smart sample
        if self.args.smart_sample:
            self.sampler = Sampler(number_of_clients=len(self.clients), greedy_mu=self.args.greedy_mu)
        self.train_tracker = [[] for c in self.clients]
        self.sample_tracker = []


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
            print('smart_ sample')
            return self.sampler.smart_sample(number_of_clients_per_round)
        return random.sample(range(len(self.clients)), number_of_clients_per_round)

    def train(self):
        args = self.args
        self.summary(0)
        self.sampled_clients_before = set()
        result = []
        for i in range(args.comm_rounds):
            clients_in_turn = self.client_sampler(args.number_of_clients_per_round, i)
            # wandb.log({"statistic data": len(self.sampled_clients_before.intersection(set(clients_in_turn))), "round": i + 1})
            self.sample_tracker.append(clients_in_turn)
            self.sampled_clients_before = set(clients_in_turn)
            print("client_indexes = " + str(sorted(clients_in_turn)))
            logging.info("client_indexes = " + str(clients_in_turn))
            weights = []
            gradients = []
            # print("test for each")
            for idx in clients_in_turn:
                w, gradient = self.clients[idx].train(self.get_model_params(), i+1)
                # with open(r"gradients.list", "wb") as output_file:
                #     pickle.dump(gradients, output_file)
                weights.append(w)
                gradients.append(gradient) # tmp = self.get_model_params()
                # self.set_model_params(updates[-1])
                # self.summary(i, use_wandb = False)
                # self.set_model_params(tmp)
            # print("####test for each END")
            if args.smart_sample:
                self.sampler.similarity_bandit(gradients, clients_in_turn)
                # np.save("./similarity/cifar10_2label100client_similarity" + str(i) +".npy", self.sampler.similarity)
            init_w = self.get_model_params_vector().clone().cuda()
            if (args.sugery == "QP"):
                gradients = self.QP_gradients_surger3(gradients, i)
                grad_global = self.weighted_average(gradients, weights)
            elif args.sugery == "init":
                fake_grad_global = self.weighted_average(gradients, weights)
                gradients = self.gradients_surgery(gradients)
                grad_global = self.weighted_average(gradients, weights)
                grad_global = grad_global/torch.norm(grad_global)*torch.norm(fake_grad_global)
            elif args.sugery == "QP_FINAL":
                gradients = self.QP_FINAL_gradients_surgery(gradients)
                grad_global = self.weighted_average(gradients, weights)
            elif args.sugery == "SIMPLE_QP":
                gradients = self.QP_SIMPLE_gradients_surgery(gradients, i)
                grad_global = self.weighted_average(gradients, weights)
            elif args.sugery == "SIMPLE_SIMPLE_QP":
                # fake_grad_global = self.weighted_average(gradients, weights)
                try:
                    gradients = self.SIMPLE_QP_SIMPLE_gradients_surgery(gradients, i)
                except:
                    print("An exception occurred")
                grad_global = self.weighted_average(gradients, weights)
                grad_global[grad_global > 0.005] = 0.005
                grad_global[grad_global < -0.005] = -0.005
                # grad_global = grad_global/torch.norm(grad_global)*torch.norm(fake_grad_global)
                # print("step size adjust")
            elif args.sugery == 'none':
                grad_global = self.weighted_average(gradients, weights)

            self.set_model_params_vector(init_w+grad_global.cuda())
            if (i % args.test_frequency == 0):
                tmp = self.summary(i+1)
                result.append((i+1, tmp))
        # self.report_clients()
        # for i in self.train_tracker:
        #     for j in i:
        #         print(j, end = ' ')
        #     print()
        # for i in self.sample_tracker:
        #     for j in i:
        #         print(j, end = ' ')
        #     print()
        return result


    def report_clients(self):
        for i, client in enumerate(self.clients):
            summary, weight = client.report_train(self.get_model_params())

            wandb.log({"Final Train/Acc": summary["train_precision"], "clientid": i})
            wandb.log({"Final Train/Loss": summary["train_loss"], "clientid": i})

            summary, weight = client.report_test(self.get_model_params())

            wandb.log({"Final Test/Acc": summary["test_precision"], "clientid": i})
            wandb.log({"Final Test/Loss": summary["test_loss"], "clientid": i})
    def QP_FINAL_gradients_surgery(self, grads):
        n = len(grads)
        grads_np = [x.cpu().detach().numpy() for x in grads]
        grads_np = np.vstack(grads_np)

        def QP(grads):
            n = grads.shape[0]
            P = np.eye(n)
            q = -np.ones([n, 1]) * 2 * 1/n

            G = -np.zeros((n, n))
            h = np.zeros([n, 1])
            # for i in range(n):
            #     for j in range(n):
            #         G[i][j] = -grads[i].dot(grads[j])

            A = np.ones((1, n))
            b = np.ones((1, 1))
            #     print(A.shape, b.shape)
            #     print(A)

            P = cvxopt.matrix(P)
            G = cvxopt.matrix(G)
            q = cvxopt.matrix(q)
            h = cvxopt.matrix(h)
            A = cvxopt.matrix(A)
            b = cvxopt.matrix(b)

            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
            q = sol['x']
            # print(sol['primal objective'])
            # print(q)
            trans = np.zeros((n, 1))
            for i in range(n):
                trans[i] = q[i]
                grads[i] = q[i] * grads[i]*n


            print(trans)
            return grads
        try:
            proj_grads = QP(grads_np)
        except:
            print("error matrix")
            return grads

        ret = []
        for i in range(n):
            ret.append(torch.from_numpy(proj_grads[i]).cuda())
        return ret


    def SIMPLE_QP_SIMPLE_gradients_surgery(self, grads, rid):
        n = len(grads)

        grads_np = [x.cpu().detach().numpy() for x in grads]
        grads_np = np.vstack(grads_np)

        def QP(grads):
            grads = np.float64(grads)
            g = grads.T
            n = grads.shape[0]
            #     print(g.shape)
            trans = np.zeros((n, n))
            for i in range(n):
                P = g.T.dot(g)
                q = -2 * g[:, i].dot(g)

                G = -P
                P = P*2
                h = np.zeros([n])
                #         print(P)
                #         print(P.shape, type(P))

                P = cvxopt.matrix(P)
                G = cvxopt.matrix(G)
                q = cvxopt.matrix(q)
                h = cvxopt.matrix(h)

                cvxopt.solvers.options['show_progress'] = False
                sol = cvxopt.solvers.qp(P, q, G, h)

                q = sol['x']
                #         print(sol['primal objective'])
                #         print(q)

                #         print(sum(fG.dot(q)<0))

                for j in range(n):
                    trans[i, j] = q[j]

            # print(trans)
            return trans.dot(grads)

        try:
            proj_grads = QP(grads_np)
        except:
            print("error matrix")
            return grads
        f = proj_grads
        # print(f[3], f[4].dot(grads.T))
        # print(f.dot(grads.T))
        a = sum(sum(grads_np.dot(grads_np.T) < 0))
        b = sum(sum(f.dot(grads_np.T) < 0))
        print("before, after:", a, b)
        wandb.log({"after/before": b / a, "round": rid + 1})
        # wandb.log({"primal objective": sol['primal objective'], "round": rid + 1})

        ret = []
        for i in range(n):
            ret.append(torch.from_numpy(proj_grads[i]))
        return ret

    def QP_SIMPLE_gradients_surgery(self, grads, rid):
        n = len(grads)
        grads_np = [x.cpu().detach().numpy() for x in grads]
        grads_np = np.vstack(grads_np)

        def QP(grads):
            g = grads.T
            n = grads.shape[0]
            # print(g.shape)
            P = np.zeros([n * n, n * n])
            q = np.zeros([n * n])
            base = g.T.dot(g)
            # print(base)
            for i in range(n):
                P[i * n: i * n + n, i * n: i * n + n] = base
                q[i * n: i * n + n] = -2 * g[:, i].T.dot(g)

            # P = P * 2
            G = np.zeros([n * n, n * n])
            h = np.zeros([n * n])

            for i in range(n):
                for j in range(n):
                    G[i * n + j, i * n: i * n + n] = -g[:, j].T.dot(g)
                    h[i * n + j] = 0

            # fG = G

            P = cvxopt.matrix(P)
            G = cvxopt.matrix(G)
            q = cvxopt.matrix(q)
            h = cvxopt.matrix(h)

            sol = cvxopt.solvers.qp(P, q, G, h)
            q = sol['x']
            # print(sol['primal objective'])
            # print(q)
            trans = np.zeros((n, n))
            # print(sum(fG.dot(q) < 0))
            for i in range(n * n):
                trans[i // n, i % n] = q[i]

            print(trans)
            return trans.dot(grads)
        proj_grads = QP(grads_np)
        f = proj_grads
        # print(f[3], f[4].dot(grads.T))
        # print(f.dot(grads.T))
        a = sum(sum(grads_np.dot(grads_np.T) < 0))
        b = sum(sum(f.dot(grads_np.T) < 0))
        print("before, after:", a, b)
        wandb.log({"after/before": b / a, "round": rid + 1})
        # wandb.log({"primal objective": sol['primal objective'], "round": rid + 1})

        ret = []
        for i in range(n):
            ret.append(torch.from_numpy(proj_grads[i]).cuda())
        return ret
    def QP_gradients_surger3(self, grads, rid):
        n = len(grads)
        grads_np = [x.cpu().detach().numpy() for x in grads]
        grads_np = np.vstack(grads_np)
        def QP(grads):
            n = grads.shape[0]
            P = np.eye(n * n)*0.5
            q = np.zeros([n * n])
            G = -np.eye(n * n)
            h = np.zeros([n * n, 1])

            As = []
            b = []
            for i in range(n):
                for j in range(n):
                    if i == j:
                        nA = np.zeros([1, n * n])
                        nA[0][i * n + j] = 1
                        b.append(1.0)
                        As.append(nA)
                    elif grads[i].dot(grads[j]) >= 0:
                        nA = np.zeros([1, n * n])
                        nA[0][i * n + j] = 1
                        As.append(nA)
                        b.append(0.0)
            A = np.vstack(As)
            b = np.stack(b)
            # print(A.shape, b.shape)
            # print(A)

            for i in range(n):
                for k in range(n):
                    nG = np.zeros([1, n * n])
                    h = np.append(h, 0)
                    for j in range(n):
                        nG[0][i * 10 + j] = -grads[j].dot(grads[k])
                    G = np.vstack([G, nG])
            #     print(G.shape, h.shape)

            P = cvxopt.matrix(P)
            G = cvxopt.matrix(G)
            q = cvxopt.matrix(q)
            h = cvxopt.matrix(h)
            A = cvxopt.matrix(A)
            b = cvxopt.matrix(b)

            sol = cvxopt.solvers.qp(P, q, G, h, A, b)
            q = sol['x']
            # print(sol['primal objective'])
            # print(q)
            trans = np.zeros((n, n))
            for i in range(n * n):
                trans[i // n, i % n] = q[i]
            f = trans.dot(grads)
            # print(f[3], f[4].dot(grads.T))
            # print(f.dot(grads.T))
            a = sum(sum(grads.dot(grads.T) < 0))
            b = sum(sum(f.dot(grads.T) < 0))
            print("before, after:", a, b)
            wandb.log({"after/before": b/a, "round": rid + 1})
            wandb.log({"primal objective": sol['primal objective'], "round":rid+1})

            # print(trans)
            return trans.dot(grads)
        proj_grads = QP(grads_np)
        ret = []
        for i in range(n):
            ret.append(torch.from_numpy(proj_grads[i]).cuda())
        return ret
    def gradients_surgery2(self, grads):
        n = len(grads)
        grads_np = [x.cpu().detach().numpy() for x in grads]
        pca = PCA(n_components=7)
        tmp = np.stack(grads_np)
        pca.fit_transform(tmp)
        print("pca variance ratio:", pca.explained_variance_ratio_, " ", sum(pca.explained_variance_ratio_))

        bases = torch.tensor(pca.components_, requires_grad=False).cuda()
        proj_grads = [bases.mm(x.unsqueeze(dim=0).T).T.mm(bases).squeeze() for x in grads]
        return proj_grads

    def gradients_surgery(self, grads):
        with torch.no_grad():
            proj_grads = [x.clone() for x in grads]
            minus_cnt = 0
            for i in range(len(grads)):
                for j in range(len(grads)):
                    if (i == j): continue
                    t = torch.dot(proj_grads[i], grads[j])
                    # print ("dot product", t.item(), (t/torch.norm(grads[i])/torch.norm(grads[j])).item())
                    if t < 0:
                        minus_cnt = minus_cnt + 1
                        proj_grads[i] = proj_grads[i] - t / (torch.norm(grads[j]) ** 2) * grads[j]
            print("minus cnt = ", minus_cnt)

            return proj_grads



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
        i = 0
        for client in self.clients:
            summary, weight = client.report_train(self.get_model_params())
            self.train_tracker[i].append(summary['train_loss'])
            i = i+1
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

