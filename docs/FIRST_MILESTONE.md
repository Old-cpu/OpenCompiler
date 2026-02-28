# 第一个里程碑：实现 Add 操作

## 目标

实现一个完整的工作流程：
1. 捕获 `x + y` 的计算图
2. 转换为 MLIR
3. 生成正确的 MLIR IR

## 学习路径

### 步骤 1：理解 PyTorch FX Graph（1-2小时）

**任务：** 运行并理解以下代码

```python
import torch
import torch.fx as fx

class SimpleAddModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y

model = SimpleAddModel()
traced = fx.symbolic_trace(model)

# 打印计算图
print(traced.graph)

# 遍历节点
for node in traced.graph.nodes:
    print(f"Node: {node.name}, Op: {node.op}, Target: {node.target}")
    print(f"  Args: {node.args}")
    print(f"  Kwargs: {node.kwargs}")
```

**问题思考：**
1. `placeholder` 节点代表什么？
2. `call_function` 节点的 `target` 是什么？
3. `output` 节点如何工作？

### 步骤 2：阅读 Buddy-MLIR 的实现（2-3小时）

**阅读顺序：**

1. **Graph 类初始化**
   - 文件：`buddy-mlir/frontend/Python/graph/graph.py`
   - 行数：102-143
   - 重点：理解 Graph 类的属性

2. **Operation 基类**
   - 文件：`buddy-mlir/frontend/Python/graph/operation.py`
   - 行数：1-100
   - 重点：Operation 的基本结构

3. **AddOp 实现**
   - 文件：`buddy-mlir/frontend/Python/ops/linalg.py`
   - 搜索：`class AddOp`
   - 重点：如何生成 MLIR IR

**记录笔记：**
- Graph 类有哪些关键属性？
- Operation 类如何设计？
- AddOp 如何生成 MLIR 操作？

### 步骤 3：设计你的 Graph 类（1-2小时）

**任务：** 创建 `OpenCompiler/graph/graph.py`

**设计要点：**

```python
class Graph:
    def __init__(self):
        # 存储操作
        self.operations = []

        # 存储输入
        self.inputs = []

        # 存储输出
        self.outputs = []

        # MLIR Context
        self.ctx = None

        # 节点映射
        self.node_map = {}
```

**问题思考：**
1. 需要哪些属性？
2. 如何存储操作？
3. 如何管理 MLIR Context？

### 步骤 4：设计 Operation 基类（1-2小时）

**任务：** 创建 `OpenCompiler/graph/operation.py`

**设计要点：**

```python
class Operation:
    def __init__(self, name, op_type):
        self.name = name
        self.op_type = op_type
        self.inputs = []
        self.outputs = []
        self.attributes = {}

    def lower_to_mlir(self, ctx):
        """生成 MLIR IR"""
        raise NotImplementedError
```

**问题思考：**
1. Operation 需要哪些基本信息？
2. 如何表示输入和输出？
3. `lower_to_mlir` 方法应该做什么？

### 步骤 5：实现 AddOp（2-3小时）

**任务：** 创建 `OpenCompiler/ops/linalg.py`

**实现步骤：**

1. **创建 AddOp 类**
```python
from mlir import ir
from mlir.dialects import arith

class AddOp(Operation):
    def __init__(self, name, inputs):
        super().__init__(name, "add")
        self.inputs = inputs

    def lower_to_mlir(self, ctx):
        # 生成 MLIR IR
        # 参考 buddy-mlir 的实现
        pass
```

2. **理解 MLIR Add 操作**
   - 阅读 MLIR Arith 方言文档
   - 理解 `arith.addf` 操作

3. **实现生成逻辑**
   - 获取输入的 MLIR 值
   - 创建 `arith.addf` 操作
   - 返回结果

### 步骤 6：连接前端和 IR 生成（2-3小时）

**任务：** 改进 `frontend.py` 和 `builder.py`

**实现步骤：**

1. **在 frontend.py 中构建 Graph**
```python
def capture(self, model, *args):
    # 捕获计算图
    # ...

    # 构建 Graph 对象
    graph = Graph()

    # 遍历节点，创建 Operation
    for node in self.captured_graph.graph.nodes:
        if node.op == "call_function":
            # 创建对应的 Operation
            pass

    return graph
```

2. **在 builder.py 中生成 MLIR**
```python
def translate(self):
    with self.ctx, ir.Location.unknown():
        module = ir.Module.create()

        # 遍历所有操作
        for op in self.graph.operations:
            op.lower_to_mlir(self.ctx)

        return module
```

### 步骤 7：测试（1-2小时）

**任务：** 创建测试用例

```python
def test_add():
    # 创建模型
    class AddModel(torch.nn.Module):
        def forward(self, x, y):
            return x + y

    model = AddModel()
    x = torch.randn(2, 3)
    y = torch.randn(2, 3)

    # 捕获计算图
    frontend = OpenCompilerFrontend()
    graph = frontend.capture(model, x, y)

    # 生成 MLIR
    builder = IRBuilder(graph)
    module = builder.translate()

    # 打印结果
    print(module)

    # 验证结果
    # 检查是否包含 arith.addf 操作
```

## 预期结果

成功后，你应该能看到类似这样的 MLIR IR：

```mlir
module {
  func.func @main(%arg0: tensor<2x3xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %0 = arith.addf %arg0, %arg1 : tensor<2x3xf32>
    return %0 : tensor<2x3xf32>
  }
}
```

## 常见问题

### Q1: MLIR Context 错误

**问题：** `RuntimeError: An MLIR function requires a Context`

**解决：** 确保在 `with self.ctx:` 或 `with ir.Location.unknown(self._ctx):` 中进行 MLIR 操作

### Q2: 类型不匹配

**问题：** 输入类型和操作类型不匹配

**解决：** 检查张量的形状和数据类型，确保一致

### Q3: 节点映射问题

**问题：** 找不到节点的输入值

**解决：** 确保 `node_map` 正确记录了每个节点的 MLIR 值

## 学习检查点

完成每个步骤后，问自己：

- [ ] 我理解了 PyTorch FX Graph 的结构吗？
- [ ] 我阅读并理解了 buddy-mlir 的相关代码吗？
- [ ] 我的设计决策有依据吗？
- [ ] 我的代码能运行吗？
- [ ] 我理解每一行代码的作用吗？

## 下一步

完成 Add 操作后，你可以：

1. 实现 Mul 操作（乘法）
2. 实现 Matmul 操作（矩阵乘法）
3. 组合多个操作，测试更复杂的模型

## 记录模板

建议创建一个学习日志：

```markdown
## 日期：2024-XX-XX

### 今天学到的内容
- ...

### 遇到的问题
- ...

### 解决方案
- ...

### 下一步计划
- ...
```

---

**记住：** 这是你第一次从零开发软件，遇到困难是正常的。关键是：
1. 📖 多阅读参考代码
2. 🤔 多思考设计决策
3. 🔨 多动手实践
4. ✅ 多写测试验证

加油！每一步都是进步！💪
