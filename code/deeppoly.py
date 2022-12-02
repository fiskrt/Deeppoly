import torch.nn.functional as F
import torch.optim as optim

from toeplitz_ops import multiple_channel_with_stride
from transformer import *
import networks

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
        self.input = inp

        self.abs_net = self.abstractize_network(orig_net)


    def forward(self, x):
        """
            Push shapes through network
        """
        return self.abs_net.forward(x)

    def verify(self):
        out = self(self.input) # Dummy call to init parameters
        #print(f'normal lb: {out.lb}')
#        print(f'bsub lb: {out.bsub_lb}')
        #print(out.bsub_ub<=out.ub)
        #print(out.bsub_lb>=out.lb)

        optimizer = optim.SGD(self.parameters(), lr=0.5)
        verified = False
        for i in range(30):
            optimizer.zero_grad()
            out = self(self.input)
#            if i ==0:
#                print(out.lb)

            loss = -out.lb.mean()
            loss.backward()
#            if i%20==0:
#                print(out.lb)

            optimizer.step()
            if (out.lb>=0).all():
                verified = True
                #return True
      #  print(out.lb)
      #  print(f'Alpha final: {list(self.parameters())[0].data}')
        
        return verified
    

    def abstractize_network(self, net):
        """
            Turn convolutions, input normalization, batch norm
            into affine layers which allows for shape propagation.
        """
        prev_shape = None
        first_conv = True
        layers = [AbstractInput(self.eps)]
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                stride = m.stride[0]
                if first_conv:
                    input_shape = self.input.shape
                else:
                    input_shape = prev_shape
                prev_shape = (m.out_channels, input_shape[1]//stride, input_shape[2]//stride)
                layers.append(AbstractAffine(multiple_channel_with_stride(kernel=m.weight.data, input_size=(m.in_channels, input_shape[1], input_shape[2]), stride=stride, padding=m.padding[0]), torch.repeat_interleave(m.bias.data, (input_shape[1]//stride) * (input_shape[2]//stride))))
            elif isinstance(m, nn.Linear):
                layers.append(AbstractAffine(m.weight.data, m.bias.data))
            elif isinstance(m, nn.ReLU):
                layers.append(AbstractReLU(m, layers[-1]))
           # elif isinstance(m, nn.Flatten):
           #     layers.append(AbstractFlatten())
            elif isinstance(m, networks.Normalization):
                layers.append(AbstractNormalize(m.mean, m.sigma))

        assert isinstance(layers[0], AbstractInput) 
        assert isinstance(layers[1], AbstractNormalize) 
#        assert isinstance(layers[2], AbstractFlatten) 
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
        return loss



if __name__=='__main__':
    from networks import get_network, get_net_name, NormalizedResnet
    from verifier import get_net
    from networks import *
    device = 'cpu'
    net = FullyConnected(device, 'mnist', 2, 1, [2, 2, 2])

    W1 = torch.Tensor([[1,1,0,0], [1,-1,0,0]])
    W2 = torch.Tensor([[1,1], [1,-1]])
    W3 = torch.Tensor([[1,0], [0,1]])

    b1 = torch.Tensor([0,0])
    #b1 = torch.Tensor([1,1.5])
    b2 = torch.Tensor([-0.5,0])
    b3 = torch.Tensor([0,0])
    my_params = [W1, b1, W2, b2, W3, b3]

    for i,param in enumerate(net.parameters()):
        param.data = my_params[i]
        print(f'set param {i}')

    inp = torch.tensor([0, 0, 250,250])
    dp = DeepPolyNet(net, inp, 1, 1)
    out = dp(inp)
    print(out.lb)
    print(out.ub)
    exit()


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
    W1 = torch.Tensor([[1,1,0,0], [1,-2,0,0]])
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

    #print(dp.backsub())

    #print('-'*20)
    #print(dp.parameters())
    #dp.loss(out)

    #dp.abs_net[0].forward()

    #dp.abs_net[1].forward(dp.abs_net[0])

    #dp.abs_net[2].forward(dp.abs_net[1])


    print('lol')
