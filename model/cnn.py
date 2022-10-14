import torch
import torch.nn as nn
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0], -1)

class CNN_OriginalFedAvg(torch.nn.Module):
    """The CNN model used in the original FedAvg paper:
    "Communication-Efficient Learning of Deep Networks from Decentralized Data"
    https://arxiv.org/abs/1602.05629.

    The number of parameters when `only_digits=True` is (1,663,370), which matches
    what is reported in the paper.
    When `only_digits=True`, the summary of returned model is

    model:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    reshape (Reshape)            (None, 28, 28, 1)         0
    _________________________________________________________________
    conv2d (Conv2D)              (None, 28, 28, 32)        832
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0
    _________________________________________________________________
    flatten (Flatten)            (None, 3136)              0
    _________________________________________________________________
    dense (Dense)                (None, 512)               1606144
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                5130
    =================================================================
    Total params: 1,663,370
    Trainable params: 1,663,370
    Non-trainable params: 0

    Args:
      only_digits: If True, uses a final layer with 10 outputs, for use with the
        digits only MNIST dataset (http://yann.lecun.com/exdb/mnist/).
        If False, uses 62 outputs for Federated Extended MNIST (FEMNIST)
        EMNIST: Extending MNIST to handwritten letters: https://arxiv.org/abs/1702.05373.
    Returns:
      A `torch.nn.Module`.
    """

    def __init__(self, input_size, num_classes):
        super(CNN_OriginalFedAvg, self).__init__()
        input_channel = input_size[0]
        self.feature = torch.nn.Sequential(
            torch.nn.Conv2d(input_channel, 32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, stride=2),
            Reshape()
        )
        self.maxpool = torch.nn.MaxPool2d(2, stride = 2)
        x = torch.zeros([1] + input_size)
        x = self.feature(x)
        num_features = x.shape[1]
        self.linear_1 = nn.Linear(num_features, 512)
        self.relu = nn.ReLU()
        self.linear_2 = nn.Linear(512, num_classes)
        # self.softmax = nn.Softmax(dim=1)
        self.__num_of_parameters = sum([p.numel() for p in self.parameters()])

    def forward(self, x):
        x = self.feature(x)
        x = self.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x

    def set_parameters_with_vector(self, param_vec):
        pointer = 0
        for param in self.parameters():
            # Ensure the parameters are located in the same device

            # The length of the parameter
            num_param = param.numel()
            # Slice the vector, reshape it, and replace the old data of the parameter
            param = param_vec[pointer:pointer + num_param].view_as(param)

            # Increment the pointer
            pointer += num_param

    def forward_with_param_vector(self, x, param_vec):
        init_param = torch.nn.utils.parameters_to_vector(self.parameters())
        torch.nn.utils.vector_to_parameters(param_vec, self.parameters())

        out = self.forward(x)

        torch.nn.utils.vector_to_parameters(init_param, self.parameters())

        return out

    def number_of_parameters(self):
        return self.__num_of_parameters

    def forward_with_param_gradients(self, x, state_dict, gradients):

        pointer = 0

        w_feature_0 = state_dict["feature.0.weight"]
        w_feature_0_len = w_feature_0.numel()
        w_feature_0_grad = gradients[pointer: pointer + w_feature_0_len].view_as(w_feature_0)
        pointer += w_feature_0_len

        b_feature_0 = state_dict["feature.0.bias"]
        b_feature_0_len = b_feature_0.numel()
        b_feature_grad = gradients[pointer: pointer + b_feature_0_len].view_as(b_feature_0)
        pointer += b_feature_0_len

        x = F.conv2d(x, w_feature_0 + w_feature_0_grad, b_feature_0 + b_feature_grad, padding=2)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        w_feature_1 = state_dict["feature.3.weight"]
        w_feature_1_len = w_feature_1.numel()
        w_feature_1_grad = gradients[pointer: pointer + w_feature_1_len].view_as(w_feature_1)
        pointer += w_feature_1_len

        b_feature_1 = state_dict["feature.3.bias"]
        b_feature_1_len = b_feature_1.numel()
        b_feature_grad = gradients[pointer: pointer + b_feature_1_len].view_as(b_feature_1)
        pointer += b_feature_1_len

        x = F.conv2d(x, w_feature_1 + w_feature_1_grad, b_feature_1 + b_feature_grad, padding=2)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(x.shape[0], -1)

        w_linear_1 = state_dict["linear_1.weight"]
        w_linear_1_len = w_linear_1.numel()
        w_linear_1_grad = gradients[pointer: pointer + w_linear_1_len].view_as(w_linear_1)
        pointer += w_linear_1_len

        b_linear_1 = state_dict["linear_1.bias"]
        b_linear_1_len = b_linear_1.numel()
        b_linear_1_grad = gradients[pointer: pointer + b_linear_1_len].view_as(b_linear_1)
        pointer += b_linear_1_len

        x = F.linear(x, w_linear_1 + w_linear_1_grad, b_linear_1 + b_linear_1_grad)
        x = F.relu(x)

        w_linear_2 = state_dict["linear_2.weight"]
        w_linear_2_len = w_linear_2.numel()
        w_linear_2_grad = gradients[pointer: pointer + w_linear_2_len].view_as(w_linear_2)
        pointer += w_linear_2_len

        b_linear_2 = state_dict["linear_2.bias"]
        b_linear_2_len = b_linear_2.numel()
        b_linear_2_grad = gradients[pointer: pointer + b_linear_2_len].view_as(b_linear_2)
        pointer += b_linear_2_len
        x = F.linear(x, w_linear_2 + w_linear_2_grad, b_linear_2 + b_linear_2_grad)
        return x

    def forward_with_param(self, x, state_dict):
        w_feature_0 = state_dict["feature.0.weight"]
        b_feature_0 = state_dict["feature.0.bias"]
        x = F.conv2d(x, w_feature_0, b_feature_0, padding = 2)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)
        
        w_feature_1 = state_dict["feature.3.weight"]
        b_feature_1 = state_dict["feature.3.bias"]
        x = F.conv2d(x, w_feature_1, b_feature_1, padding = 2)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size = 2, stride = 2)
        x = x.view(x.shape[0], -1)
        
        w_linear_1 = state_dict["linear_1.weight"]
        b_linear_1 = state_dict["linear_1.bias"]
        x = F.linear(x, w_linear_1, b_linear_1)
        x = F.relu(x)

        w_linear_2 = state_dict["linear_2.weight"]
        b_linear_2 = state_dict["linear_2.bias"]
        x = F.linear(x, w_linear_2, b_linear_2)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools

__all__ = ["ResNet9"]

class Mul(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight
    def __call__(self, x):
        return x*self.weight


def batch_norm(num_channels, bn_bias_init=None, bn_bias_freeze=False,
               bn_weight_init=None, bn_weight_freeze=False):
    m = nn.BatchNorm2d(num_channels)
    if bn_bias_init is not None:
        m.bias.data.fill_(bn_bias_init)
    if bn_bias_freeze:
        m.bias.requires_grad = False
    if bn_weight_init is not None:
        m.weight.data.fill_(bn_weight_init)
    if bn_weight_freeze:
        m.weight.requires_grad = False

    return m

#Network definition
class ConvBN(nn.Module):
    def __init__(self, do_batchnorm, c_in, c_out, bn_weight_init=1.0, pool=None, **kw):
        super().__init__()
        self.pool = pool
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1,
                              padding=1, bias=False)
        if do_batchnorm:
            self.bn = batch_norm(c_out, bn_weight_init=bn_weight_init, **kw)
        self.do_batchnorm = do_batchnorm
        self.relu = nn.ReLU(True)

    def forward(self, x):
        if self.do_batchnorm:
            out = self.relu(self.bn(self.conv(x)))
        else:
            out = self.relu(self.conv(x))
        if self.pool:
            out = self.pool(out)
        return out

    def prep_finetune(self, iid, c_in, c_out, bn_weight_init=1.0, pool=None, **kw):
        self.bn.bias.requires_grad = False
        self.bn.weight.requires_grad = False
        layers = [self.conv]
        for l in layers:
            for p in l.parameters():
                p.requires_grad = True
        return itertools.chain.from_iterable([l.parameters() for l in layers])

class Residual(nn.Module):
    def __init__(self, do_batchnorm, c, **kw):
        super().__init__()
        self.res1 = ConvBN(do_batchnorm, c, c, **kw)
        self.res2 = ConvBN(do_batchnorm, c, c, **kw)

    def forward(self, x):
        return x + F.relu(self.res2(self.res1(x)))

    def prep_finetune(self, iid, c, **kw):
        layers = [self.res1, self.res2]
        return itertools.chain.from_iterable([l.prep_finetune(iid, c, c, **kw) for l in layers])

class BasicNet(nn.Module):
    def __init__(self, do_batchnorm, channels, weight,  pool, num_classes, initial_channels=3, new_num_classes=None, **kw):
        super().__init__()
        self.new_num_classes = new_num_classes
        self.prep = ConvBN(do_batchnorm, initial_channels, channels['prep'], **kw)

        self.layer1 = ConvBN(do_batchnorm, channels['prep'], channels['layer1'],
                             pool=pool, **kw)
        self.res1 = Residual(do_batchnorm, channels['layer1'], **kw)

        self.layer2 = ConvBN(do_batchnorm, channels['layer1'], channels['layer2'],
                             pool=pool, **kw)

        self.layer3 = ConvBN(do_batchnorm, channels['layer2'], channels['layer3'],
                             pool=pool, **kw)
        self.res3 = Residual(do_batchnorm, channels['layer3'], **kw)

        self.pool = nn.MaxPool2d(4)
        self.linear = nn.Linear(channels['layer3'], num_classes, bias=False)
        self.classifier = Mul(weight)

    def forward(self, x):
        out = self.prep(x)
        out = self.res1(self.layer1(out))
        out = self.layer2(out)
        out = self.res3(self.layer3(out))

        out = self.pool(out).view(out.size()[0], -1)
        out = self.classifier(self.linear(out))
        return out

    def finetune_parameters(self, iid, channels, weight, pool, **kw):
        #layers = [self.prep, self.layer1, self.res1, self.layer2, self.layer3, self.res3]
        self.linear = nn.Linear(channels['layer3'], self.new_num_classes, bias=False)
        self.classifier = Mul(weight)
        modules = [self.linear, self.classifier]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = True
        return itertools.chain.from_iterable([m.parameters() for m in modules])
        """
        prep = self.prep.prep_finetune(iid, 3, channels['prep'], **kw)
        layer1 = self.layer1.prep_finetune(iid, channels['prep'], channels['layer1'],
                             pool=pool, **kw)
        res1 = self.res1.prep_finetune(iid, channels['layer1'], **kw)
        layer2 = self.layer2.prep_finetune(iid, channels['layer1'], channels['layer2'],
                             pool=pool, **kw)
        layer3 = self.layer3.prep_finetune(iid, channels['layer2'], channels['layer3'],
                             pool=pool, **kw)
        res3 = self.res3.prep_finetune(iid, channels['layer3'], **kw)
        layers = [prep, layer1, res1, layer2, layer3, res3]
        parameters = [itertools.chain.from_iterable(layers), itertools.chain.from_iterable([m.parameters() for m in modules])]
        return itertools.chain.from_iterable(parameters)
        """

class ResNet9(nn.Module):
    def __init__(self, do_batchnorm=False, channels=None, weight=0.125, pool=nn.MaxPool2d(2),
                 extra_layers=(), res_layers=('layer1', 'layer3'), **kw):
        super().__init__()
        self.channels = channels or {'prep': 64, 'layer1': 128,
                                'layer2': 256, 'layer3': 512}
        self.weight = weight
        self.pool = pool
        print(f"Using BatchNorm: {do_batchnorm}")
        self.n = BasicNet(do_batchnorm, self.channels, weight, pool, **kw)
        self.kw = kw

    def forward(self, x):
        return self.n(x)

    def finetune_parameters(self):
        return self.n.finetune_parameters(self.iid, self.channels, self.weight, self.pool, **self.kw)

class CIFAR10_CNN(nn.Module):
    def __init__(self, dropout):
        super(CIFAR10_CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[0]),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[1]),
            nn.Flatten()
        )

        self.clf = nn.Sequential(
            nn.Linear(5 * 5 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout[2]),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        # 50x3x32x32
        x = self.conv(x)
        x = self.clf(x)
        return x

class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(784, 64),
            nn.ReLU(inplace = True),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = torch.reshape(x, [-1, 784])
        x = self.shared_mlp(x)
        return x

class Femnist_CNN(nn.Module):
    def __init__(self):
        super(Femnist_CNN, self).__init__()
        self.shared_con = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3,3)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            # nn.Dropout()
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p = 0.5),
            nn.Flatten()
        )
        self.shared_fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 512),
            nn.ReLU(inplace = True),
            # nn.Dropout(p = 0.9),
            nn.Linear(512, 62)
        )
    def forward(self, x):
        x = self.shared_con(x)
        x = self.shared_fc(x)
        return x

class CIFAR100_CNN(nn.Module):
    def __init__(self, dropout):
        super(CIFAR100_CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[0]),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout[1]),
            nn.Flatten()
        )

        self.clf = nn.Sequential(
            nn.Linear(5 * 5 * 64, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout[2]),
            nn.Linear(1024, 100)
        )

    def forward(self, x):
        # 50x3x32x32
        x = self.conv(x)
        x = self.clf(x)
        return x

class HAR_MLP(nn.Module):
    def __init__(self, input_dim = 561, output_dim = 6):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, 250)
        self.hidden_fc = nn.Linear(250, 100)
        self.output_fc = nn.Linear(100, output_dim)

    def forward(self, x):

        # x = [batch size, height, width]

        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        # x = [batch size, height * width]

        h_1 = F.relu(self.input_fc(x))

        # h_1 = [batch size, 250]

        h_2 = F.relu(self.hidden_fc(h_1))

        # h_2 = [batch size, 100]

        y_pred = self.output_fc(h_2)

        # y_pred = [batch size, output dim]

        return y_pred

# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn

__all__ = ['AlexNet', 'alexnet']


class AlexNet(nn.Module):
  def __init__(self, classes=100):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.classifier = nn.Sequential(
        nn.Dropout(),
        nn.Linear(256 * 1 * 1, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Linear(4096, classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x


def alexnet(**kwargs):
  r"""AlexNet model architecture from the
  `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
  """
  model = AlexNet(**kwargs)
  return model

def main():
    print('debug')
    CNN_OriginalFedAvg([28, 28, 3], 10)
    f = torch.nn.Conv2d(3, 32, kernel_size=5, padding=2)




if __name__ == '__main__':
    main()