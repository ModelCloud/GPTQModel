import torch


class HookedLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        # avoid calling super().__init__() as it would allocate memory baased on in/out features
        torch.nn.Module.__init__(self)
        self.in_features = in_features
        self.out_features = out_features
            
        self.forward_hook = None

    @staticmethod
    def replace_linear_with_hooked_linear(module):
        for name, child in module.named_children():
            if isinstance(child, torch.nn.Linear):
                setattr(module, name, HookedLinear.from_linear(child))
            else:
                HookedLinear.replace_linear_with_hooked_linear(child)

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
