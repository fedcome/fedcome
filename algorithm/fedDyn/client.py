import copy
import torch
from utils.client import client
from sklearn.model_selection import train_test_split
from utils.data_process import data_to_dataloader


class client(client):
    def __init__(self, model, data, args):
        self.model = copy.deepcopy(model)
        # self.params = model.flat_w
        self.train_data = data[0]
        self.test_data = data[1]
        self.batchsize = args.batchsize
        # self.build_train_test(args)
        self.build_dataloader(args.batchsize)
        self.init_prev_grads()
        self.args = args

    def init_prev_grads(self):
        param = torch.nn.utils.parameters_to_vector(self.model.parameters())
        self.prev_grads = torch.zeros_like(param).cuda().requires_grad_(False)
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
        return copy.deepcopy(torch.nn.utils.parameters_to_vector(self.model.parameters()).detach())

    def set_model_params_vector(self, param):
        return torch.nn.utils.vector_to_parameters(param, self.model.parameters())

    # def get_model_params(self):
    #     return copy.deepcopy(self.params)
    # def set_model_params(self, flat_w):
    #     self.params = copy.deepcopy(flat_w.detach())

    # def get_model_params(self):
    #     return copy.deepcopy(self.params)

    # def set_model_params(self, flat_w):
    #     self.params = copy.deepcopy(flat_w.detach())

    # def get_model_params_vector(self):
    #     return copy.deepcopy(self.params)

    # def set_model_params_vector(self, flat_w):
    #     self.params = copy.deepcopy(flat_w.detach())

    def train(self,avg_mdl_param, alpha_coef, hist_params_diff, round_idx):
        # self.set_model_params(avg_mdl_param)
        self.set_model_params_vector(avg_mdl_param)
        device = self.args.device
        epoch = self.args.epoch
        # alpha = self.args.alpha_coef

        model = self.model

        # params = self.get_model_params().requires_grad_()
        # init = copy.deepcopy(avg)
        model.train()
        model = model.to(device)
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr*(self.args.lr_decay**round_idx), weight_decay = alpha_coef + self.args.weight_decay)
        print('lr = ', self.args.lr*(self.args.lr_decay**round_idx))
        hist_params_diff = hist_params_diff.cuda()
        avg_mdl_param = avg_mdl_param.cuda()
        for i in range(epoch):
            for (x, y) in self.train_dataloader:
                x, y = x.to(device), y.to(device)
                # model.zero_grad()
                log_probs = model.forward(x)
                class_loss = criterion(log_probs, y)

                # Get linear penalty on the current parameter estimates
                local_par_list = None
                for param in model.parameters():
                    if not isinstance(local_par_list, torch.Tensor):
                    # Initially nothing to concatenate
                        local_par_list = param.reshape(-1)
                    else:
                        local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)
            
                loss_algo = alpha_coef * torch.sum(local_par_list * (-avg_mdl_param + hist_params_diff))
            
                loss = class_loss + loss_algo

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # self.prev_grads = copy.deepcopy((self.prev_grads - alpha*(params - init)).detach())

        # self.set_model_params_vector(params)



        return (len(self.train_data), self.get_model_params_vector())

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
            pred = model.forward(x)
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

