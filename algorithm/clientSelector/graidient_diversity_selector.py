import random
import copy
import math
import numpy as np


class Sampler:
    def __init__(self, number_of_clients, greedy_mu = 0.7):
        self.number_f_clients = number_of_clients
        self.similarity = np.zeros([number_of_clients, number_of_clients])
        self.greedy_mu = greedy_mu
        print('greedy_mu = ', greedy_mu)


    def smart_sample(self, number_of_clients_per_round, round_idx = 0):
        n, m = self.similarity.shape

        def calc(sol):
            ans = 0
            ones = []
            for i in range(n):
                if sol[i] == 1:
                    ones.append(i)
            # for j in range(m):
            #     for i in range(n):
            #         ans += sol[i]*sol[j]*self.similarity[i][j]
            for j in ones:
                for i in ones:
                    ans += self.similarity[i][j]
            return ans

        def random_Evolution(sol, sol_ans):
            sol = copy.deepcopy(sol)
            cnt = 0
            ones = []
            zeros = []
            for i in range(n):
                if sol[i] == 1:
                    ones.append(i)
                else:
                    zeros.append(i)
            aim1 = random.sample(ones, 2)
            aim0 = random.sample(zeros, 2)

            for x in aim1:
                sol[x] = 0
            
            for y in aim0:
                sol[y] = 1

            # print('sum sol', sum(sol))
            # while cnt < 2:
            #     x = int(random.randint(0, n-1))
            #     y = int(random.randint(0, n-1))
            #     if (sol[x] ^ sol[y]) == 1:
            #         sol[x], sol[y] = sol[y], sol[x]
            #         cnt = cnt + 1
            return sol, calc(sol)

        def simulateAnneal(select_num):
            t = 10000
            init = random.sample(range(n), select_num)
            sol = [0 for i in range(n)]

            for i in init:
                sol[i] = 1
            sol_ans = calc(sol)

            global_ans = select_num**2
            global_sol = sol
            while t > 0.001:
                new_sol, new_ans = random_Evolution(sol, sol_ans)
                # new_ans = calc(new_sol)
                if (new_ans < global_ans):
                    global_ans, global_sol = new_ans, copy.deepcopy(new_sol)
                else:
                    # delta = calc(new_sol) - calc(sol)
                    delta = new_ans - sol_ans
                    if -delta / t > math.log(random.random()):
                        sol, sol_ans = copy.deepcopy(new_sol), new_ans
                t *= 0.97
            for i in range(1000):
                new_sol, new_ans = random_Evolution(sol, sol_ans)
                if (new_ans < global_ans):
                    global_ans, global_sol = new_ans, copy.deepcopy(new_sol)
                # calc(new_sol)
            return global_ans, global_sol

        # exploration_num = int(self.greedy_mu * (0.8**(round_idx//50))*number_of_clients_per_round)
        exploration_num = int(self.greedy_mu*number_of_clients_per_round)
        explorition_num = number_of_clients_per_round - exploration_num

        ga, gs = simulateAnneal(exploration_num)
        print("ga ", ga)
        exploration_candidate = []
        ret = []
        for i in range(len(gs)):
            if (gs[i] == 0):
                exploration_candidate.append(i)
            else:
                ret.append(i)

        rest = random.sample(exploration_candidate, explorition_num)
        ret = ret + rest
        return ret

    def similarity_bandit(self, grads, clients_in_turn):
        self.gamma = 0.5
        gamma = self.gamma
        grads_np = [x.cpu().detach().numpy() for x in grads]
        grads_np = np.vstack(grads_np)
        grads_np_norm = np.linalg.norm(grads_np, axis=1)
        n, m = grads_np.shape
        for i in range(n):
            grads_np[i] /= (grads_np_norm[i] + 1e-8)
        covar = grads_np.dot(grads_np.T)
        covar[covar > 1] = 1

        n = len(clients_in_turn)
        for i in range(n):
            for j in range(n):
                x = clients_in_turn[i]
                y = clients_in_turn[j]
                self.similarity[x][y] = gamma*self.similarity[x][y] + (1-gamma)*covar[i][j]
        print("for debug break point")

