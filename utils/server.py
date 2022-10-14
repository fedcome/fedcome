from abc import ABC, abstractmethod
import copy


class server(ABC):
    @abstractmethod
    def __init__(self, model):
        self.model = copy.deepcopy(model)

    @abstractmethod
    def get_model_params(self):
        pass

    @abstractmethod
    def set_model_params(self, model_parameters):
        pass

    @abstractmethod
    def train(self, train_data, args=None):
        pass

