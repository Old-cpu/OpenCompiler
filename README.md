# OpenCompiler

<div align="center">

**一个基于MLIR的AI编译器前端框架**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.11-green.svg)](https://www.python.org/)
[![MLIR](https://img.shields.io/badge/MLIR-21.1.8+-orange.svg)](https://mlir.llvm.org/)

[English](#english) | [中文文档](#中文文档)

### 注意本文是由AI生成的，内容可能误差
</div>

---

## 中文文档

### 项目简介

OpenCompiler 是一个基于 MLIR（Multi-Level Intermediate Representation）的 AI 编译器前端框架，旨在将 PyTorch 模型转换为 MLIR 中间表示。该项目采用清晰的分层架构设计，支持 TOSA 和 Linalg 两种 MLIR 方言，适合学习 AI 编译器开发和 MLIR 技术。

### 核心特性

- **分层架构设计**：Frontend → Graph → MLIR 三层架构，职责清晰
- **丰富的算子支持**：支持 84 个 Linalg 算子和 3 个 TOSA 算子
- **PyTorch 集成**：通过 PyTorch Dynamo 和 FX Graph 捕获计算图
- **多方言支持**：支持 TOSA、Linalg、Func 等 MLIR 方言
- **易于扩展**：模块化设计，便于添加新算子和优化 Pass
- **完善文档**：包含详细的设计文档和开发指南

### 项目结构

```
OpenCompiler/
├── docs/                          # 文档目录
│   ├── CLASS_DESIGN_DISCUSSION.md # 类设计讨论
│   ├── DEVELOPMENT_GUIDE.md       # 开发指南
│   └── FIRST_MILESTONE.md         # 第一个里程碑
├── example/                       # 示例代码
│   ├── demo1/                     # 简单模型示例
│   │   ├── Model.py              # PyTorch 模型定义
│   │   └── import-Model.py       # 模型导入示例
│   └── demo2/                     # MNIST 示例
│       ├── SimpleMNISTModel.py   # MNIST 模型
│       ├── import-SimpleMNISTModel.py
│       └── Model_test.py         # 模型训练脚本
├── OpenCompiler/                  # 核心源代码
│   ├── frontend/                  # 前端模块
│   │   └── frontend.py           # PyTorch 图捕获
│   ├── graph/                     # 图表示模块
│   │   ├── graph.py              # Graph 类
│   │   ├── operation.py          # Operation 定义
│   │   └── type.py               # 类型系统
│   └── ops/                       # 算子实现
│       ├── tosa.py               # TOSA 方言算子
│       ├── linalg.py             # Linalg 方言算子
│       ├── func.py               # Func 方言算子
│       └── utils.py              # 工具函数
├── pixi.toml                      # 依赖配置
└── README.md                      # 项目说明
```

### 快速开始

#### 环境要求

- macOS ARM64 平台
- Pixi 包管理器

#### 安装步骤

1. **安装 Pixi**（如果尚未安装）
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

2. **克隆项目**
```bash
git clone <repository-url>
cd OpenCompiler
```

3. **安装依赖**
```bash
pixi install
```

#### 运行示例

**示例 1：简单模型**
```bash
cd example/demo1
pixi run python import-Model.py
```

**示例 2：MNIST 模型**
```bash
cd example/demo2
pixi run python import-SimpleMNISTModel.py
```

### 使用方法

#### 基本用法

```python
import torch
from OpenCompiler.frontend.frontend import OpenCompilerFrontend
from OpenCompiler.ops import tosa

# 定义 PyTorch 模型
class MyModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y

# 创建模型实例
model = MyModel()
x = torch.randn(2, 3)
y = torch.randn(2, 3)

# 使用 OpenCompiler 捕获计算图
compiler = OpenCompilerFrontend(
    primary_registry=tosa.ops_registry,
    verbose=True
)
graphs = compiler.importer(model, x, y)

# 打印捕获的图
for graph in graphs:
    print(f"Graph: {graph}")
    for node in graph.body:
        print(node)
```

#### 支持的算子

**TOSA 方言算子**：
- `add` - 张量加法
- `slice` - 切片操作

**Linalg 方言算子**（部分列表）：
- **基础运算**：add, mul, div, sub, neg, matmul, pow
- **形状操作**：view, reshape, squeeze, unsqueeze, transpose
- **神经网络层**：embedding, linear, max_pool2d, avg_pool2d
- **激活函数**：relu, silu, softmax, log_softmax
- **归约操作**：mean, sum, max, min, cumsum
- **其他操作**：cat, split, index, gather, scatter, sort

完整算子列表请参考 `OpenCompiler/ops/linalg.py`。

### 架构设计

#### 三层架构

```
┌─────────────────────────────────────┐
│  Frontend Layer (前端层)            │
│  - PyTorch Dynamo 图捕获            │
│  - FX Graph 解析                    │
│  - 节点转换                         │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  Graph Layer (图表示层)             │
│  - 独立的图表示                      │
│  - Operation 管理                   │
│  - 图优化接口                        │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  MLIR Layer (MLIR 生成层)           │
│  - TOSA/Linalg 方言生成             │
│  - MLIR IR 构建                     │
│  - 类型映射                         │
└─────────────────────────────────────┘
```

#### 核心类说明

**OpenCompilerFrontend**：前端类，负责捕获 PyTorch 模型
- 使用 `torch.compile` 和 Dynamo 捕获计算图
- 解析 FX Graph 节点
- 构建 Graph 对象

**Graph**：图表示类，存储计算图结构
- 管理操作节点列表
- 维护节点查找表
- 提供图操作接口

**Operation**：操作节点基类
- 存储操作属性（名称、输入、输出）
- 维护节点依赖关系
- 支持多种操作类型

### 开发指南

#### 添加新算子

1. **在 `operation.py` 中定义操作类**
```python
class MyNewOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ElementwiseType
```

2. **在 `tosa.py` 或 `linalg.py` 中实现算子**
```python
def my_new_op(node: MyNewOp, symbol_table):
    # 获取输入
    input1 = symbol_table.get((str(node.args[0]), 0))
    # 生成 MLIR 操作
    # ...
    return op
```

3. **注册算子**
```python
ops_registry = {
    "MyNewOp": my_new_op,
}
```

#### 运行测试

```bash
# 运行 demo1
cd example/demo1 && python import-Model.py

# 运行 demo2
cd example/demo2 && python import-SimpleMNISTModel.py
```

### 文档资源

- [开发指南](docs/DEVELOPMENT_GUIDE.md) - 详细的开发路线图和技术要点
- [第一个里程碑](docs/FIRST_MILESTONE.md) - 实现 Add 操作的完整教程
- [类设计讨论](docs/CLASS_DESIGN_DISCUSSION.md) - Graph 和 IRBuilder 的设计分析
- [设计澄清](DESIGN_CLARIFICATION.md) - 数据流和实现细节说明

### 依赖项

- **Python** 3.11.*
- **MLIR** >= 21.1.8
- **PyTorch** >= 2.9.1 (CPU)
- **NumPy** >= 2.4.2
- **torchvision** >= 0.24.1

### 项目状态

当前版本：v0.1.0

**已实现功能**：
- ✅ PyTorch 模型图捕获
- ✅ Graph 图表示
- ✅ TOSA 方言算子（3个）
- ✅ Linalg 方言算子（84个）
- ✅ Func 方言算子（4个）
- ✅ 基础示例代码

**开发中功能**：
- 🔄 图优化 Pass
- 🔄 更多算子支持
- 🔄 性能优化
- 🔄 测试框架完善

### 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

### 许可证

本项目采用 Apache License 2.0 许可证 - 详见 [LICENSE](LICENSE) 文件

### 致谢

本项目参考了以下优秀项目：
- [Buddy-MLIR](https://github.com/buddy-compiler/buddy-mlir) - MLIR 学习项目
- [torch-mlir](https://github.com/llvm/torch-mlir) - PyTorch 到 MLIR 的编译器
- [MLIR](https://mlir.llvm.org/) - LLVM MLIR 框架

### 联系方式

作者：Old_cpu  
邮箱：1471913775@qq.com

---

## English

### Introduction

OpenCompiler is an AI compiler frontend framework based on MLIR (Multi-Level Intermediate Representation), designed to convert PyTorch models into MLIR intermediate representation. The project adopts a clear layered architecture design, supporting both TOSA and Linalg MLIR dialects, making it suitable for learning AI compiler development and MLIR technology.

### Key Features

- **Layered Architecture**: Frontend → Graph → MLIR three-layer architecture with clear responsibilities
- **Rich Operator Support**: 84 Linalg operators and 3 TOSA operators
- **PyTorch Integration**: Capture computation graphs via PyTorch Dynamo and FX Graph
- **Multi-Dialect Support**: TOSA, Linalg, Func and other MLIR dialects
- **Easy to Extend**: Modular design for adding new operators and optimization passes
- **Comprehensive Documentation**: Detailed design documents and development guides

### Quick Start

#### Requirements

- macOS ARM64 platform
- Pixi package manager

#### Installation

```bash
# Install Pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Clone repository
git clone <repository-url>
cd OpenCompiler

# Install dependencies
pixi install
```

#### Run Examples

```bash
# Example 1: Simple model
cd example/demo1
pixi run python import-Model.py

# Example 2: MNIST model
cd example/demo2
pixi run python import-SimpleMNISTModel.py
```

### Usage

```python
import torch
from OpenCompiler.frontend.frontend import OpenCompilerFrontend
from OpenCompiler.ops import tosa

# Define PyTorch model
class MyModel(torch.nn.Module):
    def forward(self, x, y):
        return x + y

# Create model instance
model = MyModel()
x = torch.randn(2, 3)
y = torch.randn(2, 3)

# Capture computation graph
compiler = OpenCompilerFrontend(
    primary_registry=tosa.ops_registry,
    verbose=True
)
graphs = compiler.importer(model, x, y)

# Print captured graph
for graph in graphs:
    print(f"Graph: {graph}")
    for node in graph.body:
        print(node)
```

### Documentation

- [Development Guide](docs/DEVELOPMENT_GUIDE.md) - Detailed development roadmap and technical points
- [First Milestone](docs/FIRST_MILESTONE.md) - Complete tutorial for implementing Add operation
- [Class Design Discussion](docs/CLASS_DESIGN_DISCUSSION.md) - Design analysis of Graph and IRBuilder

### Dependencies

- **Python** 3.11.*
- **MLIR** >= 21.1.8
- **PyTorch** >= 2.9.1 (CPU)
- **NumPy** >= 2.4.2

### License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

This project references the following excellent projects:
- [Buddy-MLIR](https://github.com/buddy-compiler/buddy-mlir)
- [torch-mlir](https://github.com/llvm/torch-mlir)
- [MLIR](https://mlir.llvm.org/)

---

<div align="center">

**如果这个项目对您有帮助，请给一个 ⭐️ Star！**

Made with ❤️ by Old_cpu

</div>
