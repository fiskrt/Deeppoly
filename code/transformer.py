"""
    These are the abstract transformers for each layer. 
    To each layer we add upperbound, lowerbound
"""

class AbstractLayer(nn.Module):
    """
        A network is a composition of layers which are functions.
        An abstract network is a composition of abstract layers 
        which are abstract transformers.
    """
    pass


class AbstractAffine(nn.Module):

    def __init__(self, layer):
        super().__init__()
        assert isinstance(layer, nn.Linear), 'not linear'
        # Each node contain a upper and lower bound: 1D array
        self.ub = None
        self.lb = None
        # Each node contain a upper and lower linear constraint:
        # In an affine layer these are the same.
        self.W_upper = layer.weight 
        self.b_upper = layer.bias
        self.W_lower = layer.weight 
        self.b_lower = layer.bias
        
    def forward(self, prev_layer):
        # Push shape from previous layer through this layer

        # For each node in this layer, we have a constraint
        # with 'in_features'+1 coefficients.
        # The coefficient matrix is simply w and b concatinated.


        w_pos = self.w >= 0.
        self.ub = (w_pos*self.w)@prev_layer.ub+(~(w_pos)*self.w)@prev_layer.lb+self.b 
        self.lb = (w_pos*self.w)@prev_layer.lb+(~(w_pos)*self.w)@prev_layer.ub+self.b 

        return 


class InputLayer(nn.Module):
    """
        Input layer have no previous layer, but the bounds are given.
        We only have to clamp these to satisify normalization.
    """

class ReLULayer(AbstractLayer):

    def __init__(self):
        self.ub = None
        self.lb = None

        self.W_upper = layer.weight 
        self.b_upper = layer.bias
        self.W_lower = layer.weight 
        self.b_lower = layer.bias
        
        # TODO: How to init alpha? random/minimum area?
        alpha = torch.zeros(a.shape, requires_grad=True)

    def forward(self, prev_layer):    
        # Before a ReLU layer we want to backsub so we maximize our chance of
        # getting a non-crossing ReLU.

        # When ReLU crossing we always have positive coef infront of constraint bounds
        # The same is true for negative and positive relu

        # There are three cases
        neg_mask = prev_layer.ub <= 0.
        pos_mask = prev_layer.lb > 0.
        cross = ~(neg_mask | pos_mask)

        # Negative case: constraints are just 0
        self.ub[neg_mask] = 0.
        self.lb[neg_mask] = 0.
        
        self.W_upper[neg_mask] = 0.
        self.W_lower[neg_mask] = 0.

        # Positive case:
        self.ub[pos_mask] = prev_layer.ub
        self.lb[pos_mask] = prev_layer.lb

        self.W_upper[pos_mask] = prev_layer.W_upper
        self.W_lower[pos_mask] = prev_layer.W_lower


        # Crossing case:
        # The ReLU upperbound will be some a^Tx+b
        # where a_i is given by u/(u-l) and b_i=-u*l/(u*l)
        a = prev_layer.ub / (prev_layer.ub-prev_layer.lb)
        b = -prev_layer.lb * a
        self.W_upper = torch.einsum('i,ij->ij', a, prev_layer.W_upper)
        self.b_upper = prev_layer.b_upper + b

        # Calculate lowerbound with alpha parameterization alpha*x
        self.W_lower = torch.einsum('i,ij->ij', alpha, prev_layer.W_lower)
        self.b_lower = prev_layer.b_upper





