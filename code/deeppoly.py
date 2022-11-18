import torch
import torch.nn.functional as F
import torch.nn as nn
from transformer import *

from networks import get_network, get_net_name, NormalizedResnet
from verifier import get_net


class DeepPolyNet(nn.Module):
    """
        Turn a network into a network that allows for bound propagation.

    """
    def __init__(self, orig_net, eps):
        super().__init__()
        self.eps = eps
        self.prev_layer = None
        self.is_residual = False

        self.abs_net = self.abstractize_network(orig_net)

    def __call__(self, x):
        return self.forward(x)


    def forward(self, x):
        """
            Push shapes through network
        """
        return self.abs_net.forward(x)


    def abstractize_network(self, net):
        """
            Turn convolutions, input normalization, batch norm
            into affine layers which allows for shape propagation.
        """
        layers = [AbstractInput(self.eps)]
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                print(m)
            elif isinstance(m, nn.Linear):
                print(m)
                layers.append(AbstractAffine(m))
            elif isinstance(m, nn.ReLU):
                print(m)
                layers.append(AbstractReLU(m, layers[-1]))
        return nn.Sequential(*layers)
                


#n = get_net('net8', get_net_name('net8'))
#dp = DeepPolyNet(n)
from networks import *
device = 'cpu'
net = FullyConnected(device, 'mnist', 2, 1, [2, 2])

print("Model's state_dict:")
#for param_tensor in net.state_dict():
#    print(param_tensor, "\t", net.state_dict()[param_tensor])

#torch.save(net.state_dict(), 'weights')
W1 = torch.Tensor([[1,1,0,0], [1,-2,0,0]])
W2 = torch.Tensor([[1,1], [-1,1]])

b1 = torch.Tensor([0,0])
b2 = torch.Tensor([0,0])
my_params = [W1, b1, W2, b2]

for i,param in enumerate(net.parameters()):
    param.data = my_params[i]
    print(param)

dp = DeepPolyNet(net, 1)
#print(dp.abs_net)

inp = torch.tensor([0.5, 0.5, 5,5])
out = dp(inp)

#dp.abs_net[0].forward()

#dp.abs_net[1].forward(dp.abs_net[0])

#dp.abs_net[2].forward(dp.abs_net[1])

print('lol')
