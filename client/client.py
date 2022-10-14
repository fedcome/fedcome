import copy
import utils.data_process

class Client:
    def __init__(self, data, model_trainer):
        # data is list of samples
        self.data = data
        self.num_data = len(data)
        self.model_trainer = copy.deepcopy(model_trainer)

        self.data_loader = None
        self.batch_size = None


