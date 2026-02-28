import torch
from OpenCompiler.frontend.frontend import OpenCompilerFrontend
from OpenCompiler.ops import tosa
from Model import MyModule

model = MyModule()
data = torch.rand(5, 4)

open_compile = OpenCompilerFrontend(primary_registry=tosa.ops_registry,
                                    verbose=True)

gm = open_compile.importer(model,data)

print("---- Success Capture Model ----")
# print(gm)
print(str(gm))

# 打印图中所有节点的详细信息
print("\n---- Graph Nodes Details ----")
if len(gm) > 0:
    graph = gm[0]  # 获取第一个图
    print(f"Total nodes in graph: {len(graph.body)}\n")
    
    # 方式1: 遍历所有节点并打印
    for node in graph.body:
        print(node)
        print()  # 空行分隔
    
    # 方式2: 通过节点名称访问特定节点
    # print("\n---- Access specific node by name ----")
    # if 'add' in graph.node_table:
    #     print(graph.node_table['add'])

