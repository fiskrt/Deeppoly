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

    def __init__(self):
        super().__init__()
        # Each node contain a upper and lower bound: 1D array
        self.ub = None
        self.lb = None
        # Each node contain a upper and lower linear constraint:
        self.uc = None
        self.lc = None
        
    def forward(self, prev_layer):
        # Push shape from previous layer through this layer


class InputLayer(nn.Module):
    """
        Clamp stuff
    """
    pass

class ReLULayer(AbstractLayer):
    pass