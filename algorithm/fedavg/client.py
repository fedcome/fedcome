import copy
import torch
from utils.client import client
from sklearn.model_selection import train_test_split
from utils.data_process import data_to_dataloader


class client(client):
    def __init__(self, model, data, args):
        self.model = copy.deepcopy(model)
        self.data = data
        self.batchsize = args.batchsize
        self.build_train_test(args)
        self.build_dataloader(args.batchsize)
        self.args = args

    def build_train_test(self, args):
        if len(self.data) == 1:
            self.train_data, self.test_data = self.data, self.data
            return
        self.train_data, self.test_data = train_test_split(self.data, train_size = 5.0/6)

    def build_dataloader(self, batchsize):
        self.train_dataloader = data_to_dataloader(self.train_data, batchsize)
        self.test_dataloader = data_to_dataloader(self.test_data, batchsize)

    def get_model_params(self):
        return copy.deepcopy(self.model.cpu().state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, server_parameters):
        self.set_model_params(server_parameters)
        device = self.args.device
        epoch = self.args.epoch

        model = self.model
        model.train()
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for i in range(epoch):
            for (x, y) in self.train_dataloader:
                x, y = x.to(device), y.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, y)
                loss.backward()
                optimizer.step()


        return (len(self.train_data), self.get_model_params())

    def test(self, dataloader):

        device = self.args.device
        model = self.model
        model.to(device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss(reduction="sum").to(device)
        summary = {}
        loss, correct, total = 0, 0, len(dataloader.dataset)

        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            prob, index = torch.max(pred, dim = 1)
            loss += criterion(pred, y).item()
            correct += (index==y).sum().item()
        return loss/total, correct/total





    def report_test(self, server_parameters):
        summary = {}
        self.set_model_params(server_parameters)
        summary["test_loss"], summary["test_precision"] = self.test(self.test_dataloader)
        return summary, len(self.test_data)

    def report_train(self, server_parameters):
        summary = {}
        self.set_model_params(server_parameters)
        summary["train_loss"], summary["train_precision"] = self.test(self.train_dataloader)
        return summary, len(self.train_data)

    def encode(self, updates, args=None):
        print("hello encode")

    def decode(self, zip_updates, args=None):
        print("hello decode")

    def aggregate(self, updates, args=None):
        print("hello aggregate")

