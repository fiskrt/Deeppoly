import torch.nn.functional as F
import torch.optim as optim

from conv2affine import conv_to_affine
from transformer import *
from networks import BasicBlock, Normalization, NormalizedResnet

class DeepPolyNet(nn.Module):
    """
        Turn a network into a network that allows for bound propagation.
    """
    def __init__(self, orig_net, inp, eps, true_label):
        super().__init__()
        self.eps = eps
        self.prev_layer = None
        self.is_residual = False
        self.true_label = true_label 
        if inp.dim() == 3:
            self.input = inp
        elif inp.dim() == 4:
            # Remove batch dimension
            self.input = inp.squeeze(0)
        else:
            assert False, 'Input shape error'

        self.abs_net = self.abstractize_network(orig_net)


    def forward(self, x):
        """
            Push shapes through network
        """
        return self.abs_net.forward(x)

    def verify(self):
        out = self(self.input) # Dummy call to init parameters
        #print(f'lb: {out.lb}')

        optimizer = optim.SGD(self.parameters(), lr=0.5)
        verified = False
        for i in range(50):
            optimizer.zero_grad()
            out = self(self.input)

        #    print(out.lb[out.lb<0])
            loss = -out.lb.mean()
            loss.backward()
#            if i%20==0:
#                print(out.lb)

            if (out.lb>=0).all():
                #print(f'Took {i} iterations')
                #print(out.lb)
                verified = True
                return True
            optimizer.step()
      #  print(out.lb)
        #print(f'Alpha final: {list(self.parameters())[0].data}')
        
        return verified
    
    def _layers_to_abstract(self, net, prev_shape, deeper=True):
        """
            Only check modules of max depth = 2. Ignore deeper ones.
        """
        layers = []
        for m in net.children():
#            if len(list(m.children())) > 0 and deeper:
            if not isinstance(m, BasicBlock):
                l, prev_shape = self._layers_to_abstract(m, prev_shape, deeper=False)
                layers.extend(l)

            if isinstance(m, nn.Conv2d):
                W, b, prev_shape = conv_to_affine(m, prev_shape)
                layers.append(AbstractAffine(W,b))
            elif isinstance(m, nn.Linear):
                layers.append(AbstractAffine(m.weight.data, m.bias.data))
            elif isinstance(m, nn.ReLU):
                layers.append(AbstractReLU())
            elif isinstance(m, Normalization):
                layers.append(AbstractNormalize(m.mean, m.sigma))
            elif isinstance(m, BasicBlock):
                path_a, block_out_shape_a = self._layers_to_abstract(m.path_a, prev_shape)
                path_b, block_out_shape_b = self._layers_to_abstract(m.path_b, prev_shape)
                assert block_out_shape_a == block_out_shape_b
                prev_shape = block_out_shape_a
                layers.append(AbstractBlock(path_a, path_b))
        return layers, prev_shape

    def abstractize_network(self, net):
        """
            Turn convolutions, input normalization, batch norm
            into affine layers which allows for shape propagation.
        """
        # keep track of the shape (num_channels, H, W)
        prev_shape = self.input.shape
        layers = [AbstractInput(self.eps)]

        ls, _ = self._layers_to_abstract(net, prev_shape)
        layers.extend(ls)

        assert isinstance(layers[0], AbstractInput) 
        assert isinstance(layers[1], AbstractNormalize) 
        assert isinstance(layers[-1], AbstractAffine), "Final layer is not affine!"
        layers[-1] = AbstractOutput(layers[-1].W, layers[-1].b, self.true_label)

        return nn.Sequential(*layers)

if __name__=='__main__':
    from networks import get_network
    n = get_network('cpu', 'net10')
    n = NormalizedResnet('cpu', n)
    inp = torch.ones((1,3,32,32))
#    print(n(inp))
    dp = DeepPolyNet(n, inp, 1, 3)
    out = dp(inp)
    print()
    exit()

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
