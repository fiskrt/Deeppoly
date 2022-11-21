import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from transformer import *



class DeepPolyNet(nn.Module):
    """
        Turn a network into a network that allows for bound propagation.

    """
    def __init__(self, orig_net, inp, eps, true_label):
        super().__init__()
        self.eps = eps
        self.prev_layer = None
        self.is_residual = False
        #print(f'orignal net outp: {orig_net(inp)}')
        #self.target = orig_net(inp).squeeze()
        self.true_label = true_label # the actual number, not the index!
        assert inp.dim() == 1, "Input is not 1d!"
        self.input = inp

        self.abs_net = self.abstractize_network(orig_net)


    def forward(self, x):
        """
            Push shapes through network
        """
        return self.abs_net.forward(x)

    def verify(self):
        self(self.input) # Dummy call to init parameters
        optimizer = optim.SGD(self.parameters(), lr=0.1)
        print(self)
        for i in range(1000):
            optimizer.zero_grad()
            out = self(self.input)
            if i ==0:
                print(out.lb)
            # change loss to sum/mean of all negative?
            loss = -out.lb[out.lb<0].mean()
            loss.backward()
            if i%300==0:
                print(out.lb)
                #print([p for p in self.parameters()])
            optimizer.step()
            if (out.lb>=0).all():
                return True
        
        return False

    def abstractize_network(self, net):
        """
            Turn convolutions, input normalization, batch norm
            into affine layers which allows for shape propagation.
        """
        layers = [AbstractInput(self.eps)]
        last = None
        #print('layers:')
        for m in net.modules():
            #print(m)
            if isinstance(m, nn.Conv2d):
                pass
            elif isinstance(m, nn.Linear):
        #        print(m)
                layers.append(AbstractAffine(m.weight.data, m.bias.data))
                last = m
            elif isinstance(m, nn.ReLU):
        #        print(m)
                layers.append(AbstractReLU(m, layers[-1]))
        
        print('-'*50)
        assert isinstance(layers[-1], AbstractAffine), "Final layer is not affine!"
        layers[-1] = AbstractOutput(layers[-1].W, layers[-1].b, self.true_label)
        return nn.Sequential(*layers)


    
    def loss(self, output):
        """
            output: tensor of lb/ub
            target: the predicted class that we want to prove robust
        """
        target = F.one_hot(self.target.argmax(keepdim=True), output.lb.shape[0]).squeeze().bool()
        lb_correct = output.lb[target]
        ub_wrong_classes = output.ub[~target]
        loss = -(lb_correct - ub_wrong_classes.max())


if __name__=='__main__':
    from networks import get_network, get_net_name, NormalizedResnet
    from verifier import get_net
    #n = get_net('net8', get_net_name('net8'))
    #dp = DeepPolyNet(n)
    from networks import *
    device = 'cpu'
    net = FullyConnected(device, 'mnist', 2, 1, [2, 2])

    #print("Model's state_dict:")
    #for param_tensor in net.state_dict():
    #    print(param_tensor, "\t", net.state_dict()[param_tensor])

    #torch.save(net.state_dict(), 'weights')
    W1 = torch.Tensor([[-0.5,1,0,0], [1,-2,0,0]])
    W2 = torch.Tensor([[1,1], [-1,1]])

    b1 = torch.Tensor([0,0])
    b2 = torch.Tensor([0,0])
    my_params = [W1, b1, W2, b2]

    for i,param in enumerate(net.parameters()):
        param.data = my_params[i]
        #print(param)


    #print(dp.abs_net)

    inp = torch.tensor([0.5, 0.5, 0,0])
    dp = DeepPolyNet(net, inp, 1, 0)
    #print('-'*20)
    #print(dp.parameters())
    print(dp)
    out = dp(inp)

    #print('-'*20)
    #print(dp.parameters())
    #dp.loss(out)

    #dp.abs_net[0].forward()

    #dp.abs_net[1].forward(dp.abs_net[0])

    #dp.abs_net[2].forward(dp.abs_net[1])


    print('lol')
