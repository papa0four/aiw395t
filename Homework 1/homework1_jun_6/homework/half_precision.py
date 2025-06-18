from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


class HalfLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """
        Implement a half-precision Linear Layer.
        Feel free to use the torch.nn.Linear class as a parent class (it makes load_state_dict easier, names match).
        Feel free to set self.requires_grad_ to False, we will not backpropagate through this layer.
        """
        # TODO: Implement me
        super().__init__(in_features, out_features, bias)
        
        self.weight.data = self.weight.data.half()
        self.weight.requires_grad_(False)
        
        if self.bias is not None:
            self.bias.data = self.bias.data.half()
            self.bias.requires_grad_(False)
        # raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hint: Use the .to method to cast a tensor to a different dtype (i.e. torch.float16 or x.dtype)
        # The input and output should be of x.dtype = torch.float32
        # TODO: Implement me            
        if x.device.type == 'cpu':
            weight_f32 = self.weight.float()
            bias_f32 = self.bias.float() if self.bias is not None else None
            return torch.nn.functional.linear(x, weight_f32, bias_f32)
        else:
            x_half = x.to(torch.float16)
            out_half = torch.nn.functional.linear(x_half, self.weight, self.bias)
            return out_half.to(x.dtype)
        # raise NotImplementedError()


class HalfBigNet(torch.nn.Module):
    """
    A BigNet where all weights are in half precision. Make sure that the normalization uses full
    precision though to avoid numerical instability.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
                torch.nn.ReLU(),
                HalfLinear(channels, channels),
            )
            # raise NotImplementedError()

        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )
        # raise NotImplementedError()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def _convert_to_half(self):
    """Convert all HalfLinear layers to half precision and disable gradients"""
    for module in self.modules():
        if isinstance(module, HalfLinear):
            module.weight.data = module.weight.data.half()
            module.weight.requires_grad_(False)
            if module.bias is not None:
                module.bias.data = module.bias.data.half()
                module.bias.requires_grad_(False)

def load(path: Path | None) -> HalfBigNet:
    # You should not need to change anything here
    # PyTorch can load float32 states into float16 models
    net = HalfBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
