"""
    These are the abstract transformers for each layer. 
    To each layer we add upperbound, lowerbound
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter, UninitializedParameter


class AbstractLayer(nn.Module):
    """
        A network is a composition of layers which are functions.
        An abstract network is a composition of abstract layers 
        which are abstract transformers.
    """
    pass


class AbstractAffine(nn.Module):

    def __init__(self, W, b):
        super().__init__()
#        assert isinstance(layer, nn.Linear), 'not linear'
        # Each node contain a upper and lower bound: 1D array
        self.ub = None
        self.lb = None
        self.input_ub = None
        self.input_lb = None
        # Each node contain a upper and lower linear constraint:
        # In an affine layer these are the same.
        self.W = W
        self.b = b
        self.W_upper = None
        self.b_upper = None
        self.W_lower = None
        self.b_lower = None
        self.n_out, self.n_in = self.W.shape
    
    def __str__(self):
        return 'AbstractAffine'
        
    def forward(self, prev_layer):
        self.prev = prev_layer
        # Push shape from previous layer through this layer
        self.input_ub = prev_layer.input_ub
        self.input_lb = prev_layer.input_lb

        # For each node in this layer, we have a constraint
        # with 'in_features'+1 coefficients.
        # The coefficient matrix is simply w and b concatinated.
        w_pos = self.W >= 0.
        self.W_upper2 = self.W.clone()
        self.W_lower2 = self.W.clone()
        self.b_upper2 = self.b.clone()
        self.b_lower2 = self.b.clone()

        self.W_upper = (w_pos*self.W)@prev_layer.W_upper+(~w_pos*self.W)@prev_layer.W_lower
        self.W_lower = (w_pos*self.W)@prev_layer.W_lower+(~w_pos*self.W)@prev_layer.W_upper
        self.b_lower = self.W@prev_layer.b_lower+self.b
        self.b_upper = self.W@prev_layer.b_upper+self.b

        # Since W_upper and W_lower contain the constraints in terms of the input layers
        # we use the input lower/upper bound. I.e this automatically does backsub to input layer. 
        wu_pos = self.W_upper >= 0.
        wl_pos = self.W_lower >= 0.
        self.ub = (wu_pos*self.W_upper)@self.input_ub+(~wu_pos*self.W_upper)@self.input_lb+self.b_upper 
        self.lb = (wl_pos*self.W_lower)@self.input_lb+(~wl_pos*self.W_lower)@self.input_ub+self.b_lower 

        return self

class AbstractOutput(AbstractAffine):
    """
        This layer combines the final affine layer with
        the output layer that does x_best-x_j
    """
    def __init__(self, W, b, pred):
        # Create matrix for x_best-x 
        M_1 = -1.*torch.eye(W.shape[0]) 
        M_2 = torch.zeros_like(M_1)
        M_2[:,pred] = 1.
        M = M_1+M_2
        M = torch.cat((M[:pred], M[pred+1:]))

        #print(f'M: {M}')
        # bias is always zero in output layer!
        super().__init__(M@W, M@b)
        #self.pred = pred 

    def __str__(self):
        return 'AbstractOutput'

    def forward(self, prev_layer):
        return super().forward(prev_layer)


class AbstractInput(nn.Module):
    """
        Input layer have no previous layer, but the bounds are given.
        We only have to clamp these to satisify normalization.
    """
    def __init__(self, eps):
        super().__init__()
        self.eps = eps
        self.n_in = None
        self.n_out = None

        self.W_lower = None 
        self.W_upper = None 
        self.b_lower = None 
        self.b_upper = None 

        self.ub = None
        self.lb = None

        self.input_ub = None
        self.input_lb = None

        self.prev = None


    def __str__(self):
        return 'AbstractInput'
    
    def forward(self, x):
        #self.x = x
        assert x.dim() == 1, 'Dimension should be one in input!'
        self.n_in =x.shape[0]
        self.n_out = self.n_in
        self.W_lower = torch.eye(self.n_in)
        self.W_upper = torch.eye(self.n_in)
        self.b_lower = torch.zeros(self.n_in)
        self.b_upper = torch.zeros(self.n_in)

        #remove
        self.W_lower2 = torch.eye(self.n_in)
        self.W_upper2 = torch.eye(self.n_in)
        self.b_upper2 = torch.zeros(self.n_in)
        self.b_lower2 = torch.zeros(self.n_in)
        
        # TODO: input x have to be normalized!!!!!!!
        self.input_ub = torch.clamp(x+self.eps, max=1.)
        self.input_lb = torch.clamp(x-self.eps, min=0.)
        self.ub = self.input_ub.clone()
        self.lb = self.input_lb.clone()

        #print(f'Inited input layer: ub/lb {self.ub}/{self.lb}')

        return self



class AbstractReLU(nn.Module):

    def __init__(self, layer, last_layer):
        super().__init__()
        self.prev = None
        self.ub = None
        self.lb = None
        self.input_ub = None
        self.input_lb = None

        self.n_in = None
        self.n_out = None

        # wrong inits?
        self.W_upper = None
        self.b_upper = None
        self.W_lower = None
        self.b_lower = None

        self.inited = False

#        self.alpha = UninitializedParameter()
        
    def __str__(self):
        return 'AbstractReLU'

    def forward(self, prev_layer):    
        self.prev = prev_layer
        # Before a ReLU layer we want to backsub so we maximize our chance of
        # getting a non-crossing ReLU.

        # When ReLU crossing we always have positive coef infront of constraint bounds
        # The same is true for negative and positive relu
        self.input_ub = prev_layer.input_ub
        self.input_lb = prev_layer.input_lb
        self.n_in = prev_layer.n_out
        self.n_out = self.n_in

        # There are three cases
        neg_mask = prev_layer.ub <= 0.
        pos_mask = prev_layer.lb >= 0.
        cross_mask = ~(neg_mask | pos_mask)

        #print(f'Number of crossing ReLUs: {cross_mask.sum()}')
        #print(f'Pos relu at positions: {pos_mask} with incoming lb/ub: {prev_layer.lb[pos_mask]}/{prev_layer.ub[pos_mask]}')
        #print(f'neg relu at positions: {neg_mask} with incoming lb/ub: {prev_layer.lb[neg_mask]}/{prev_layer.ub[neg_mask]}')
        #print(f'Crossing relu at positions: {cross_mask} with incoming lb/ub: {prev_layer.lb[cross_mask]}/{prev_layer.ub[cross_mask]}')

        # Negative case: constraints are just 0
        self.ub = torch.zeros(self.n_out)
        self.lb = torch.zeros(self.n_out)
        
        self.W_upper = torch.zeros((self.n_out, prev_layer.W_upper.shape[1]))
        self.b_upper = torch.zeros(self.n_out)
        self.W_lower = torch.zeros((self.n_out, prev_layer.W_lower.shape[1]))
        self.b_lower = torch.zeros(self.n_out)

        self.W_upper2 = torch.zeros((self.n_in, self.n_in))
        self.W_lower2 = torch.zeros((self.n_in, self.n_in))
        self.b_upper2 = torch.zeros(self.n_in)
        self.b_lower2 = torch.zeros(self.n_in)

        # Positive case:
        self.ub[pos_mask] = prev_layer.ub[pos_mask]
        self.lb[pos_mask] = prev_layer.lb[pos_mask]

        self.W_upper[pos_mask] = prev_layer.W_upper[pos_mask]
        self.b_upper[pos_mask] = prev_layer.b_upper[pos_mask]
        self.W_lower[pos_mask] = prev_layer.W_lower[pos_mask]
        self.b_lower[pos_mask] = prev_layer.b_lower[pos_mask]

        self.W_upper2 = torch.eye(self.n_in)
        self.W_lower2 = torch.eye(self.n_in)

        # Crossing case: 
        if cross_mask.any():
            # TODO: How to init alpha? random/minimum area?
            # make sure between 0,1 softmax?
            # TODO: When initing this, it may depend on alpha from previous layer
            # should it then not be a parameter?
            # TODO: Init all alphas since during training the crossing relus can change place
            # and change in count. But the first layer alphas are always the same? 
            with torch.no_grad():
                if not self.inited:
                    self.inited = ~self.inited
                    alpha = torch.zeros(self.n_out)
                    #alpha[prev_layer.ub[cross_mask] > -prev_layer.lb[cross_mask]] = 1.
        #            self.alpha.materialize(alpha.shape)
        #            self.alpha.data = alpha #= Parameter(self.alpha)
        #            self.alpha = torch.tensor(alpha, requires_grad=True)
                    self.alpha = Parameter(alpha)
                    print(f'alpha from trans: {cross_mask.sum()}')
                if self.inited:
                    self.alpha.data = self.alpha.clamp(0,1)

    #        self.alpha = torch.zeros(last_layer.b.shape, requires_grad=True)
            # The ReLU upperbound will be some a^Tx+b
            # where a_i is given by u/(u-l) and b_i=-u*l/(u*l)
            a = prev_layer.ub[cross_mask] / (prev_layer.ub[cross_mask]-prev_layer.lb[cross_mask])
            b = -prev_layer.lb[cross_mask] * a
            self.W_upper[cross_mask] = torch.einsum('i,ij->ij', a, prev_layer.W_upper[cross_mask])
            self.b_upper[cross_mask] = prev_layer.b_upper[cross_mask] + b

            # Calculate lowerbound with alpha parameterization alpha*x
            self.W_lower[cross_mask] = torch.einsum('i,ij->ij', self.alpha[cross_mask], prev_layer.W_lower[cross_mask])
            self.b_lower[cross_mask] = prev_layer.b_upper[cross_mask]

            # After a ReLU backsub cannot help us, so we only need the ub/lb from previous layer
            self.ub[cross_mask] = prev_layer.ub[cross_mask]
            # Lowerbound after a ReLU is always alpha*x
            self.lb[cross_mask] = self.alpha[cross_mask]*prev_layer.lb[cross_mask]

            a2 = torch.zeros(self.n_in)
            a2[cross_mask] = a
            alpha2 = torch.zeros(self.n_in)
            alpha2[cross_mask] = self.alpha[cross_mask]

            self.W_upper2[cross_mask] = torch.diag(a2)[cross_mask]
            self.b_upper2[cross_mask] = b
            self.W_lower2[cross_mask] = torch.diag(alpha2)[cross_mask]
            self.b_lower2[cross_mask] = 0. 

        return self

