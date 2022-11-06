"""
    These are the 
    To each layer we add upperbound, lowerbound
"""

class AbstractLayer(nn.Module):
    """
        A network is a composition of layers which are functions.
        An abstract network is a composition of abstract layers 
        which are abstract transformers.
    """
    pass


class AffineLayer(nn.Module):

    def __init__(self, layer):
        super().__init__()
        # Each node contain a upper and lower bound: 1D array
        self.ub = None
        self.lb = None
        # Each node contain a upper and lower linear constraint:
        self.w = layer.weight 
        self.b = layer.bias
        
    def forward(self, prev_layer):
        # Push shape from previous layer through this layer

        # For each node in this layer, we have a constraint
        # with 'in_features'+1 coefficients.
        # The coefficient matrix is simply w and b concatinated.

        w_pos = self.w >= 0.
        self.ub = (w_pos*self.w)@prev_layer.ub+(~(w_pos)*self.w)@prev_layer.lb+self.b 
        self.lb = (w_pos*self.w)@prev_layer.lb+(~(w_pos)*self.w)@prev_layer.ub+self.b 




class InputLayer(nn.Module):
    """
        Input layer have no previous layer, but the bounds are given.
        We only have to clamp these to satisify normalization.
    """
    pass

class ReLULayer(AbstractLayer):
    pass