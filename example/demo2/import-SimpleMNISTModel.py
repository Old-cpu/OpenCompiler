from SimpleMNISTModel import SimpleMNISTModel
import torch
from OpenCompiler.frontend.frontend import OpenCompilerFrontend
from OpenCompiler.ops import tosa

model = SimpleMNISTModel()
model.load_state_dict(torch.load("/Users/old_people/test/OpenCompiler/example/demo2/simple_mnist_model.pth"))
model.eval()
data = torch.rand(1, 28, 28)

open_compiler = OpenCompilerFrontend(primary_registry=tosa.ops_registry,
                                    verbose=True)

gm = open_compiler.importer(model, data)

# print("---- MLIR Graph ----\n")

print(str(gm))
# print(gm.code)