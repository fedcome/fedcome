
from client.client import Client
import copy
import numpy as np


class Server:
    def __init__(self, model_trainer, server_data = None):
        self.training_data = server_data

        self.client_list = [Client(data, model_trainer) for data in clients_data]
        self.num_clients = len(clients_data)

        self.model_trainer = copy.deepcopy(model_trainer)

    def client_sampler(self, num_clients_per_round):
        return self.client_list[np.random.sample(range(self.num_clients), num_clients_per_round)]

    def train(self, args):
        num_clients_per_round = args.number_of_clients
        clients_involved = self.client_sampler(num_clients_per_round)
        updates = []
        for client in clients_involved:
            updates.append(client.train(args))
        self.model_trainer.aggregrate(updates)



    # def set_clients(self, clients_data):

