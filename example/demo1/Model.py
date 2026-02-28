import torch
# import torch.fx

class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        w = self.linear.weight
        x = x + w
        return x

# m = MyModule()
# gm = torch.fx.symbolic_trace(m)
# print(gm.graph)
# print(gm.code)
