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
    def __init__(self):
        super().__init__()

    def backsub(self, grad=True):
        #with torch.no_grad()
        layer = self
        W_upper = layer.W_upper2.clone()
        W_lower = layer.W_lower2.clone()
        b_upper = layer.b_upper2.clone()
        b_lower = layer.b_lower2.clone()
        #print('---------------backsub---------------------')
        best_lb = self.lb
        best_ub = self.ub
        while layer.prev:
            prev = layer.prev
            
            M_u = W_upper >= 0
            M_l = W_lower >= 0
            ub = M_u*W_upper@prev.ub + ~M_u*W_upper@prev.lb + b_upper
            lb = M_l*W_lower@prev.lb + ~M_l*W_lower@prev.ub + b_lower
            best_lb = torch.max(best_lb, lb)
            best_ub = torch.min(best_ub, ub)


            M_u = W_upper >= 0
            M_l = W_lower >= 0
            b_upper = M_u*W_upper@prev.b_upper2 + ~M_u*W_upper@prev.b_lower2 + b_upper
            b_lower = M_l*W_lower@prev.b_lower2 + ~M_l*W_lower@prev.b_upper2 + b_lower
#            b_lower = W_lower@prev.b_lower2 + b_lower
            W_upper = (M_u*W_upper)@prev.W_upper2 + (~M_u*W_upper)@prev.W_lower2
            W_lower = (M_l*W_lower)@prev.W_lower2 + (~M_l*W_lower)@prev.W_upper2
            #print(f'At layer {layer}:') 
            #print(f'W_upper: {W_upper.data}')
            #print(f'b_upper: {b_upper.data}')
            #print(f'W: {layer.W_upper2.data}')
            #print(f'W_lower: {W_lower}')
            
            #continously check if we have good enough bounds, i.e if lb>0 then we can stop?


            #print(f'At layer {layer} the upper constraints: {W_upper}')
            layer = prev
        # We reached the input layer and we can now calc the ub/lb with the
        # input weights
        M_u = W_upper >= 0
        M_l = W_lower >= 0
        ub = M_u*W_upper@layer.ub + ~M_u*W_upper@layer.lb + b_upper
        lb = M_l*W_lower@layer.lb + ~M_l*W_lower@layer.ub + b_lower
        best_lb = torch.max(best_lb, lb)
        best_ub = torch.min(best_ub, ub)
        #print('---------------backsubend---------------------')
        #print(f'best_lb: {best_lb}')
        #print(f'best_ub: {best_ub}')
        return best_ub, best_lb
        

class AbstractAffine(AbstractLayer):

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
        self.b_upper = (w_pos*self.W)@prev_layer.b_upper+(~w_pos*self.W)@prev_layer.b_lower + self.b

        self.W_lower = (w_pos*self.W)@prev_layer.W_lower+(~w_pos*self.W)@prev_layer.W_upper
        self.b_lower = (w_pos*self.W)@prev_layer.b_lower+(~w_pos*self.W)@prev_layer.b_upper + self.b

        # Since W_upper and W_lower contain the constraints in terms of the input layers
        # we use the input lower/upper bound. I.e this automatically does backsub to input layer. 
        # @@@@@ change to W_upper2 etc.
        wu_pos = self.W_upper2 >= 0.
        wl_pos = self.W_lower2 >= 0.
        # @@@@@ Lowerbound in terms of input layer
        #self.ub = (wu_pos*self.W_upper)@self.input_ub+(~wu_pos*self.W_upper)@self.input_lb+self.b_upper 
        #self.lb = (wl_pos*self.W_lower)@self.input_lb+(~wl_pos*self.W_lower)@self.input_ub+self.b_lower 
        # @@@@@ bounds in term of previous layer
        self.ub = (wu_pos*self.W_upper2)@prev_layer.ub+(~wu_pos*self.W_upper2)@prev_layer.lb+self.b_upper2 
        self.lb = (wl_pos*self.W_lower2)@prev_layer.lb+(~wl_pos*self.W_lower2)@prev_layer.ub+self.b_lower2 

        bsub = self.backsub()
        self.bsub_ub = bsub[0]
        self.bsub_lb = bsub[1]
        if (bsub[0] == self.ub).all() and (bsub[1] == self.lb).all():
            print(f'BACKSUB and ub/lb are same in {self}')
        else:
            print(f'BACKSUB NOT same in {self}!!')

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
        l = super().forward(prev_layer)
        print('calling backsub on last layer')
        ub, lb = self.backsub()
        print(f'backsub ub: {ub}')
        print(f'backsub lb: {lb}')
        return l


class AbstractInput(AbstractLayer):
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
        # Remove batch dimension
        x = x.squeeze(0)
        assert x.dim() == 3, 'Dimension should be 3 in input!'
        # Cheat and just flatten the the image HxW to HW immediately
        x = x.flatten(start_dim=1)
        self.input_ub = torch.clamp(x+self.eps, min=0., max=1.)
        self.input_lb = torch.clamp(x-self.eps, min=0., max=1.) # should be zero

        self.ub = self.input_ub.clone()
        self.lb = self.input_lb.clone()

        #self.x = self.norm_layer(x)
        #self.x = self.x.flatten()
        #self.n_in = self.x.shape[0]
        #self.n_out = self.n_in
        #self.W_lower = torch.eye(self.n_in)
        #self.W_upper = torch.eye(self.n_in)
        #self.b_lower = torch.zeros(self.n_in)
        #self.b_upper = torch.zeros(self.n_in)

        ##remove
        #self.W_lower2 = torch.eye(self.n_in)
        #self.W_upper2 = torch.eye(self.n_in)
        #self.b_upper2 = torch.zeros(self.n_in)
        #self.b_lower2 = torch.zeros(self.n_in)
        
        # TODO: UNCOMENT@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # TODO: input x have to be normalized!!!!!!!

        #print(f'Inited input layer: ub/lb {self.ub}/{self.lb}')

        return self

class AbstractFlatten(AbstractLayer):
    """
        Takes BxWxH-> W*H

        TODO: set prev=None so this is seen as 'first layer'?

        Flatten might not copy the value, is this bad?
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, prev_layer):
        # want backsub to stop here?
        assert False
        self.prev = None
        self.input_ub = prev_layer.input_ub.flatten()
        self.input_lb = prev_layer.input_lb.flatten()

        self.n_out = self.input_ub.shape[0]

        self.W_upper = prev_layer.W_upper.flatten()
        self.W_lower = prev_layer.W_lower.flatten()

        self.b_upper = prev_layer.b_upper.flatten()
        self.b_lower = prev_layer.b_lower.flatten()

        self.ub = prev_layer.ub.flatten()
        self.lb = prev_layer.lb.flatten()

        self.W_upper2 = self.W_upper.clone()
        self.W_lower2 = self.W_lower.clone()
        self.b_upper2 = self.b_upper.clone()
        self.b_lower2 = self.b_lower.clone()

        return self


class AbstractNormalize(AbstractLayer):
    def __init__(self, mean, sigma):
        super().__init__()
        self.prev = None

        self.mean = mean
        self.sigma = sigma

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
    
    def forward(self, prev_layer):
        # set this as first layer?
        self.prev = None
        #self.input_ub = prev_layer.input_ub
        #self.input_lb = prev_layer.input_lb

        #self.n_in = prev_layer.n_out
        #self.n_out = self.n_in
        prev_shape = prev_layer.lb.shape
        if prev_shape[0] == 1:
            m = self.mean.squeeze().unsqueeze(0)
            s = self.sigma.squeeze().unsqueeze(0)
            self.ub = (1/s)*prev_layer.ub.squeeze(0)-(m/s)
            self.lb = (1/s)*prev_layer.lb.squeeze(0)-(m/s)
            self.W_upper = torch.eye(self.ub.shape[0])
            self.W_lower = torch.eye(self.ub.shape[0])
            self.b_upper = torch.zeros(self.ub.shape[0])
            self.b_lower = torch.zeros(self.ub.shape[0])
            assert self.ub.dim() == 1
        elif prev_shape[0] == 3:
            m = self.mean.squeeze()
            s = self.sigma.squeeze()
            self.ub = ((1/s)[:, None]*prev_layer.ub - (m/s)[:, None]).flatten()
            self.lb = ((1/s)[:, None]*prev_layer.lb - (m/s)[:, None]).flatten()
            self.W_upper = torch.eye(self.ub.shape[0])
            self.W_lower = torch.eye(self.ub.shape[0])
            self.b_upper = torch.zeros(self.ub.shape[0])
            self.b_lower = torch.zeros(self.ub.shape[0])
        else:
            assert False, 'Number of channels must be 1 or 3!'

        #I = torch.eye(prev_shape[1])
        ## store just the diagonals C x WH, otherwise too big
        ## then in Flatten we unroll and then matrix=diag(unrolled)
        #self.W_upper = torch.einsum('ij,k->kij', I, 1.0/s)
        #self.W_lower = self.W_upper.clone() 
        #self.b_upper = -(m/s)[:,None]*torch.ones((prev_shape[0],prev_shape[1]))
        #self.b_lower = self.b_upper.clone()

        ## just change ub/lb to new

        ## W upper and b is just same numbers so can we use some broadcasting to
        ## calculate ub/lb. Combine this layer and flatten?

        ## store only diagonal in W_upper, we can
        #self.ub = self.W_upper
        #self.ub = self.W_upper@prev_layer.ub+self.b_upper
        #self.lb = self.W_lower@prev_layer.lb+self.b_lower

#        self.input_lb = 

        self.W_upper2 = self.W_upper
        self.W_lower2 = self.W_lower
        self.b_upper2 = self.b_upper
        self.b_lower2 = self.b_lower

        return self
        



class AbstractReLU(AbstractLayer):

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
        print('calling backsub before ReLU layer')
        ub, lb = prev_layer.backsub()
        if (ub == prev_layer.ub).all() and (lb == prev_layer.lb).all():
            print(f'BACKSUB and ub/lb are same before {self}')
        else:
            print(f'BACKSUB NOT same before {self}!!')
        prev_layer.ub = ub
        prev_layer.lb = lb
        #print(f'backsub ub: {ub}')
        #print(f'backsub lb: {lb}')
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


        arr = torch.zeros(self.n_in)
        arr[pos_mask] = 1
        self.W_upper2[pos_mask] = torch.diag(arr)[pos_mask]
        self.W_lower2[pos_mask] = torch.diag(arr)[pos_mask]

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
                    alpha = 0.* torch.ones(self.n_out)
                    alpha[prev_layer.ub > -prev_layer.lb] = 1.
                    #print(f'inited alpha to: {alpha}')
        #            self.alpha.materialize(alpha.shape)
        #            self.alpha.data = alpha #= Parameter(self.alpha)
        #            self.alpha = torch.tensor(alpha, requires_grad=True)
                    self.alpha = Parameter(alpha)
                    print(f'Number of crossing ReLUs: {cross_mask.sum()}')
                # remove if here? useless?
                if self.inited:
                    self.alpha.data = self.alpha.clamp(0,1)

    #        self.alpha = torch.zeros(last_layer.b.shape, requires_grad=True)
            # The ReLU upperbound will be some a^Tx+b
            # where a_i is given by u/(u-l) and b_i=-u*l/(u*l)
            a = prev_layer.ub[cross_mask] / (prev_layer.ub[cross_mask]-prev_layer.lb[cross_mask])
            b = -prev_layer.lb[cross_mask] * a
            self.W_upper[cross_mask] = torch.einsum('i,ij->ij', a, prev_layer.W_upper[cross_mask])
            self.b_upper[cross_mask] = a*prev_layer.b_upper[cross_mask] + b

            # Calculate lowerbound with alpha parameterization alpha*x
            self.W_lower[cross_mask] = torch.einsum('i,ij->ij', self.alpha[cross_mask], prev_layer.W_lower[cross_mask])
            #self.b_lower[cross_mask] = prev_layer.b_lower[cross_mask]


            # After a ReLU backsub cannot help us, so we only need the ub/lb from previous layer
            self.ub[cross_mask] = prev_layer.ub[cross_mask]
            # Lowerbound after a ReLU is always alpha*x
            #self.lb[cross_mask] = self.alpha[cross_mask]*(self.W_lower[cross_mask]@prev_layer.lb[cross_mask])
            # when alpha>0 so we get lower constrain x2>= alpha*x, we must backsub to get the real lowerbound
            #wl_pos = self.W_lower>= 0
            #temp_lb = (wl_pos*self.W_lower)@self.input_lb+(~wl_pos*self.W_lower)@self.input_ub+self.b_lower 
            #self.lb[cross_mask] = temp_lb[cross_mask] # dont have to mult by alpha since, i use W_lower
            # still incorrect, if alpha=0, the lb should be 0

            # Or should it just be 0, since even if we have nonzero alpha, we know that output
            # from relu is always >= 0


            a2 = torch.zeros(self.n_in)
            a2[cross_mask] = a
            alpha2 = torch.zeros(self.n_in)
            alpha2[cross_mask] = self.alpha[cross_mask]

            self.W_upper2[cross_mask] = torch.diag(a2)[cross_mask]
            self.b_upper2[cross_mask] = b
            self.W_lower2[cross_mask] = torch.diag(alpha2)[cross_mask]
            self.b_lower2[cross_mask] = 0. 
        
        bsub = self.backsub()
        if (bsub[0] == self.ub).all() and (bsub[1] == self.lb).all():
            print(f'BACKSUB and ub/lb are same in {self}')
        else:
            print(f'BACKSUB NOT same in {self}!!')

        return self

if __name__ == '__main__':
    inp = torch.tensor([1, 2, 0, -3, 0])
    abs_inp = AbstractInput(eps=1)
    abs_relu = AbstractReLU(abs_inp, None)

    res1 = abs_inp(inp)
    res2 = abs_relu(res1)

    print()

