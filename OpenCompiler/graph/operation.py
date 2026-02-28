from enum import Enum
from typing import Dict, Optional, List, Tuple

class OpType(Enum):

    BroadcastType = 0
    ElementwiseType = 1
    ReshapeType = 2
    SliceLikeType = 3
    ReduceType = 4
    ConcatType = 5
    PlaceholderType = 6
    GetItemType = 7
    Unfusable = 8

class Op:
    def __init__(self) -> None:
        self._name = None
        self._arguments = []
        self._args_index = []
        self._keyword_arguments = {}
        self._parents: List[str] = []
        self._children: List[str] = []
        self._tensor_meta: Dict = {}
        self._op_type: OpType = None

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, new_name):
        self._name = new_name

    @property
    def tensor_meta(self):
        return self._tensor_meta
    
    @tensor_meta.setter
    def tensor_mata(self, new_tensor_meta):
        self._tensor_meta = new_tensor_meta



    def add_argument(self, arg, arg_index=0):
        """
        向操作节点添加输入参数
        """
        self._arguments.append(arg)
        self._args_index.append(arg_index)

    def add_parent(self, parent: str):
        """
        向操作节点添加一个父节点的名字
        """
        self._parents.append(parent)

    def add_childern(self, child):
        """
        向操作节点添加一个儿子节点的名字
        """
        self._children.append(child)
    
    def __str__(self):
        """
        返回节点的详细信息字符串
        """
        return f"""====================Graph Node====================
        Node: {self._name}
        Type: {self._op_type}
        Arguments: {self._arguments}
        Parents: {self._parents}
        Children: {self._children}"""



class PlaceholderOp(Op):
    def __init__(self) -> None:
        super().__init__()
        self._op_type = OpType.PlaceholderType

class OutputOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.GetItemType

class AddOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.BroadcastType

class SliceOp(Op):
    def __init__(self):
        super().__init__()
        self._op_type = OpType.ReshapeType

