import copy
from regex import W
import torch
import numpy as np
from utils.client import client
from sklearn.model_selection import train_test_split
from utils.data_process import data_to_dataloader


class client(client):
    def __init__(self, model, data, args):
        self.model = copy.deepcopy(model)
        # a = torch.nn.utils.parameters_to_vector(self.model.parameters())
        # self.local_c = torch.zeros_like(a).cuda()
        self.train_data = data[0]
        self.test_data = data[1]
        # self.batchsize = args.batchsize
        self.batchsize = args.batchsize
        # self.build_train_test(args)
        self.build_dataloader(self.batchsize)
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

    def get_model_params_vector(self):
        return torch.nn.utils.parameters_to_vector(self.model.parameters())

    def set_model_params_vector(self, param):
        return torch.nn.utils.vector_to_parameters(param, self.model.parameters())

    def datasize(self):
        return len(self.train_data)

    def train(self, server_parameters, global_update_last, alpha, local_update_last, hist_i, round_idx):
        server_parameters = server_parameters.cuda()
        self.set_model_params_vector(server_parameters)
        device = self.args.device
        epoch = self.args.epoch

        state_update_diff = torch.tensor(-local_update_last + global_update_last, dtype = torch.float32, device = device)

        model = self.model
        model = model.to(device)
        model.train()
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr*(self.args.lr_decay**round_idx))
        K = np.ceil(len(self.train_data)/self.args.batchsize)*self.args.epoch
        # for i in range(epoch):
        #     K = K + 1
        for i in range(epoch):
            for (x, y) in self.train_dataloader:
                x, y = x.to(device), y.to(device)
                log_probs = model(x)
                loss = criterion(log_probs, y)
                local_param_list = None
                for param in model.parameters():
                    if not isinstance(local_param_list, torch.Tensor):
                        # Initially nothing to concatenate
                        local_param_list = param.reshape(-1)
                    else:
                        local_param_list = torch.cat((local_param_list, param.reshape(-1)), 0)

                t = (local_param_list - (server_parameters - hist_i))
                loss_cp = alpha/2 * torch.sum(t*t)
                # loss_cp = 0
                scafford_loss = torch.sum(local_param_list*state_update_diff.cuda())
                # scafford_loss = torch.sum(local_param_list*state_update_diff.cuda())
                # scafford_loss = 0
                # cur = torch.nn.utils.parameters_to_vector(model.parameters())
                loss = loss + loss_cp + scafford_loss
                optimizer.zero_grad()
                loss.backward()
                # scafford_loss.backward()
                # loss_cp.backward()
                # loss.backward()
                # loss_all = loss + scafford_loss + loss_cp
                # loss_all.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()
                if (self.args.batch_mode):
                    break
            if (self.args.batch_mode):
                break
        # with torch.no_grad():
        #     after = torch.nn.utils.parameters_to_vector(model.parameters())
        #     ret_c = self.local_c - global_c + 1/(K*self.args.lr)*(init - after) - self.local_c
        #     ret_c = copy.deepcopy(ret_c)
        #     self.local_c = self.local_c - global_c + 1/(K*self.args.lr)*(init - after)


        return (len(self.train_data), self.get_model_params_vector(), K)

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

