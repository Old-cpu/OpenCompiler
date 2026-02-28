from typing import Any, List, Optional, Dict
from types import FunctionType
from .type import *
from .operation import *

import torch

class Graph:
    def __init__(
        self,
        inputs: List[TensorMeta],
        fake_params: List[TensorMeta],
        ops_register: dict,
        func_name: str,
        device: DeviceType = DeviceType.CPU,
        verbose = False,
        enable_external_calls: bool = False,
        ) -> None:
        '''
        初始化图

        Args:
            inputs: List[TensorMeta]
                以 TensorMeta 对象表示的模型输入。
            fake_params: List[TensorMeta]
                以 TensorMeta 对象表示的虚拟参数。
            ops_registry: dict
                图中算子的降级策略（lowering strategy）。
            func_name: str
                MLIR 模块的函数名称。
            enable_external_calls: bool
                启用外部函数调用支持（例如 oneDNN 等）。
        '''

        self._body = [] # 代表计算图的主干，模型执行的操作序列
        self.node_table: Dict[str, Op] = {}
        self._inputs = inputs
        self._fake_params = fake_params
        self._ops_register = ops_register
        self._func_name = func_name
        self._device = device
        self._verbose = verbose
        self._enable_external_calls = enable_external_calls
        self._fake_params = fake_params,
        self._params_ref = None,

    @property
    def body(self):
        return self._body
    
    @body.setter
    def body(self, new_body):
        self._body = new_body


    def add_node(self, node: Op):
        self._body.append(node)
        self.node_table[node.name] = node

