import torch.nn as nn

class ANN_cycle(nn.Module):
    """
    ANN cycle that consists of linear layer, RelU and dropout ones.
    
    Args:
        in_features: number of input neurons in the current ANN cycle
        out_features: number of output neurons in the current ANN cycle
    """
    def __init__(self, in_features, out_features, dropout_val=0.3) -> None:
        super(ANN_cycle, self).__init__()
        self.block = nn.Sequential(nn.Linear(in_features, out_features),
                                   nn.BatchNorm1d(out_features),
                                   nn.ReLU(),
                                   nn.Dropout(dropout_val))
        
    def forward(self, x):
        return self.block(x)

class ANN_Blocks(nn.Module):
    """
    A stack of ANN cycles.

    Each ANN cycle consists of a linear layer, a ReLU activation function, and a dropout layer.

    Args:
        None
    """
    def __init__(self, in_features, dropout_val=0.3, out_features=16) -> None:
        super(ANN_Blocks, self).__init__()
        self.block = nn.Sequential(ANN_cycle(in_features, 512, dropout_val),
                                 ANN_cycle(512, 256, dropout_val),
                                 ANN_cycle(256, 128, dropout_val),
                                 ANN_cycle(128, 64, dropout_val),
                                 ANN_cycle(64, 32, dropout_val),
                                 ANN_cycle(32, out_features, dropout_val))

    def forward(self, x):
        return self.block(x)