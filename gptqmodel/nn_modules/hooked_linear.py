import torch


class HookedConv1D(torch.nn.Conv1d):
    def __init__(self, nf: int, nx: int) -> None:
        torch.nn.Module.__init__(self)
        self.nf = nf
        self.nx = nx
        self.forward_hook = None

    @staticmethod
    def from_conv1d(conv1d: torch.nn.Conv1d):
        custom_conv1d = HookedConv1D(conv1d.nf, conv1d.nx)
        custom_conv1d.weight = conv1d.weight
        custom_conv1d.bias = conv1d.bias
        return custom_conv1d    

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.forward_hook:
            self.forward_hook(self, input, output)
        return output


class HookedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        # avoid calling super().__init__() as it would allocate memory baased on in/out features
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
            
        self.forward_hook = None

    @staticmethod
    def from_linear(linear: torch.nn.Linear):
        custom_linear = HookedLinear(linear.in_features, linear.out_features)
        custom_linear.weight = linear.weight
        custom_linear.bias = linear.bias
        return custom_linear

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output = super().forward(input)
        if self.forward_hook:
            self.forward_hook(self, input, output)
        return output


def replace_linear_with_hooked_linear(module):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            setattr(module, name, HookedLinear.from_linear(child))
        elif isinstance(child, torch.nn.Conv1d):
            setattr(module, name, HookedConv1D.from_conv1d(child))
        else:
            replace_linear_with_hooked_linear(child)
