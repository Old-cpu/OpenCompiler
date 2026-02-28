import torch
import torch._dynamo as dynamo
from torch._functorch.aot_autograd import aot_module_simplified

from typing import Any, Tuple, Optional
import operator
# from ..ir.builder import IRBuilder

from ..graph.graph import Graph
from ..ops.tosa import ops_registry as tosa_ops_registry
from ..graph.operation import *

from ..graph.type import *

class OpenCompilerFrontend:
    def __init__(
        self, 
        func_name: str = "forward",
        primary_registry: Optional[dict] = None,
        aot_autograd_decomposition: Optional[dict] = None,
        verbose=False,
        enable_external_calls: bool = False,
        ) -> None:

        #清除 Dynamo 的所有内部状态和缓存
        dynamo.reset()

        if primary_registry is None:
            primary_registry = {}

        self._func_name = func_name
        self._imported_graphs = []
        self._imported_params = {}
        self._aot_autograd_decomposition = aot_autograd_decomposition
        # self._model_config = type("Config", (),)
        self._verbose = verbose
        self._enable_external_calls = enable_external_calls
        self._ops_registry = {}
        self._ops_registry.update(tosa_ops_registry)

        self._ops_map = {
            "output": OutputOp,
            "placeholder": PlaceholderOp,
            "add.Tensor": AddOp,
            "slice.Tensor": SliceOp,
        }

        self.captured_graph = None
        self.tensor_inputs = None

    @property
    def imported_graphs(self):
        """Returns the imported buddy graphs after compilation."""
        return self._imported_graphs
    
    @property
    def imported_params(self):
        """Returns the imported model params after compilation."""
        return self._imported_params

    def _torch_dtype_translate(self, dtype):
        '''数据类型转换'''
        match dtype:
            case "torch.int8":
                return TensorDType.Int8
            case "torch.float32":
                return TensorDType.Float32
            case "torch.float64":
                return TensorDType.Float64
            case "torch.bool":
                return TensorDType.Bool
            case _:
                raise NotImplementedError(f"Unsupported dtype: {dtype}")
            

    def _create_node(
        self,
        gm_node_name: str,
        node_name: str,
        node_input: Tuple,
        node_user: List[str],
        node_output_shape: list = [],
        node_output_dtype: TensorDType = None,
        node_kwargs: Optional[Dict] = None,
    ):
        op_class = self._ops_map[gm_node_name]
        oc_node = op_class()
        oc_node._name = node_name
        if gm_node_name == "output":
            for input_arg in node_input[0]:
                oc_node.add_argument(str(input_arg))
            return oc_node
        
        def _add_arg_and_parents(arg):
            if isinstance(arg, torch.fx.Node):
                oc_node.add_argument(str(arg))
                oc_node.add_parent(str(arg))
            elif isinstance(arg, torch.dtype):
                oc_node.add_argument(self._torch_dtype_translate(str(arg)))
            elif isinstance(arg, (list, tuple)):
                for item in arg:
                    if isinstance(item, torch.fx.Node):
                        oc_node.add_argument(str(item))
                oc_node.add_argument(arg)
            else:
                oc_node.add_argument(arg)
            return arg
        
        for input_arg in node_input:
            _add_arg_and_parents(input_arg)
        for user in node_user:
            oc_node.add_childern(user)
        if node_kwargs is None:
            node_kwargs = {}

        oc_node._keyword_arguments.update(node_kwargs)
        oc_node._tensor_meta["shape"] = node_output_shape
        oc_node._tensor_meta["dtype"] = node_output_dtype
        return oc_node
            

    def _compiler_fx(self, gm: torch.fx.GraphModule, inputs: List[torch.Tensor]) -> Any:
        inputs_pos = []
        params_pos = []
        buffers_pos = []

        for i, node in enumerate(gm.graph.nodes):
            if i >= len(inputs):
                break
            if not str(node).startswith("l_self"):
                inputs_pos.append(i)
            elif "buffer" in str(node):
                buffers_pos.append(i)
            else:
                params_pos.append(i)

        params_flat = [inputs[i] for i in params_pos + buffers_pos]

        if self._verbose:
            print("Grapht in tabular from:")
            gm.graph.print_tabular()

        def _compiler(_gm: torch.fx.GraphModule, _inputs: List[torch.Tensor]):

            '''在这里构建函数输入元数据'''
            func_inputs = []
            for i in inputs_pos:
                inp = _inputs[i]
                inp_shape = inp.shape
                inp_dtype = inp.dtype
                func_inputs.append(TensorMeta(inp_shape, inp_dtype))

            '''构建参数元数据'''
            fake_params = []
            for param in params_flat:
                param_dtype = self._torch_dtype_translate(str(param.dtype))
                fake_params.append(TensorMeta(param.shape, param_dtype))

            '''自定义Graph对象'''
            graph = Graph(
                func_inputs,
                fake_params,
                self._ops_registry,
                self._func_name,
                DeviceType.CPU,
                self._verbose,
                self._enable_external_calls,
            )

            graph._fake_params = params_flat

            '''按类型分组并重新排序 FX 图的节点'''
            param_nodes = []
            buffers_nodes = []
            input_nodes = []
            other_nodes = []

            for i, node in enumerate(_gm.graph.nodes):
                if i in params_pos:
                    param_nodes.append(node)
                elif i in buffers_pos:
                    buffers_nodes.append(node)
                elif i in inputs_pos:
                    input_nodes.append(node)
                else:
                    other_nodes.append(node)

            # 按照参数、缓冲区、输入、其他的顺序合并
            gm_nodes = param_nodes + buffers_nodes + input_nodes + other_nodes

            '''遍历每个 FX 节点，转换为 OC 节点'''
            for gm_node in gm_nodes:
                node_users = []
                for user in gm_node.users.keys():
                    node_users.append(str(user))
                
                if gm_node.op == "placeholder":
                    node_dtype = self._torch_dtype_translate(
                        str(gm_node.meta["tensor_meta"].dtype)
                    )

                    oc_node = self._create_node(
                        gm_node.op,
                        gm_node.name,
                        gm_node.args,
                        node_users,
                        gm_node.meta["tensor_meta"].shape,
                        node_dtype,
                    )

                elif gm_node.op == "output":
                    oc_node = self._create_node(
                        gm_node.op,
                        gm_node.name,
                        gm_node.args,
                        node_users,
                    )

                elif gm_node.target is operator.getitem:
                    node_dtype = self._torch_dtype_translate(
                        str(gm_node.meta["tensor_meta"].dtype)
                    )
                    oc_node = self._create_node(
                        str(gm_node.target.__name__),
                        gm_node.name,
                        gm_node.args,
                        node_users,
                        gm_node.meta["tensor_meta"].shape,
                        node_dtype,
                    )

                elif gm_node.op == "get_attr":
                    if "_tensor_constant" in gm_node.name:
                        import re

                        stack_trace = gm_node.meta.get("stack_trace") or ""
                        match = re.search(
                            r"torch\.tensor\(([-+]?\d+(\.\d+)?), dtype=[a-zA-Z]+\)",
                            stack_trace,
                        )
                        value = None
                        if match:
                            value = float(match.group(1))
                        if value is None:
                            val = gm_node.meta.get("val")
                            if isinstance(val, torch.Tensor):
                                if val.numel() != 1:
                                    raise NotImplementedError(
                                        "_tensor_constant only supports scalar tensors"
                                    )
                                value = val.item()
                            elif isinstance(val, (int, float)):
                                value = val
                        if value is None:
                            raise NotImplementedError(
                                "Unsupported _tensor_constant format"
                            )
                        gm_node.insert_arg(len(gm_node.args), value)
                        val = gm_node.meta.get("val")
                        node_shape = val.shape
                        node_dtype = self._torch_dtype_translate(str(val.dtype))
                        oc_node = self._create_node(
                            "_tensor_constant",
                            gm_node.name,
                            gm_node.args,
                            node_users,
                            node_shape,
                            node_dtype,
                            node_kwargs=gm_node.kwargs,
                        )

                else:
                    tensor_meta = gm_node.meta.get("tensor_meta")
                    val = gm_node.meta.get("val")
                    # num_returns = len(gm_node.target._schema.returns)
                    num_returns = (
                        len(val)
                        if isinstance(val, (list, tuple))
                        else len(gm_node.target._schema.returns)
                    )
                    if num_returns == 1:
                        if tensor_meta is None:
                            if isinstance(val, torch.Tensor):
                                node_dtype = self._torch_dtype_translate(
                                    str(val.dtype)
                                )
                                node_shape = val.shape
                            elif str(gm_node.target) == "aten.unbind.int":
                                input_node = gm_node.args[0]
                                dim = (
                                    gm_node.args[1]
                                    if len(gm_node.args) > 1
                                    else 0
                                )
                                input_meta = input_node.meta.get("tensor_meta")
                                input_val = input_node.meta.get("val")
                                if input_meta is None:
                                    if not isinstance(input_val, torch.Tensor):
                                        raise RuntimeError(
                                            "Missing input meta for aten.unbind.int"
                                        )
                                    input_shape = list(input_val.shape)
                                    input_dtype = input_val.dtype
                                else:
                                    input_shape = list(input_meta.shape)
                                    input_dtype = input_meta.dtype
                                if dim < 0:
                                    dim += len(input_shape)
                                length = input_shape[dim]
                                if length < 0:
                                    raise RuntimeError(
                                        "Dynamic unbind dimension not supported"
                                    )
                                out_shape = tuple(
                                    input_shape[:dim] + input_shape[dim + 1 :]
                                )
                                node_shape = tuple([out_shape] * length)
                                node_dtype = tuple(
                                    [
                                        self._torch_dtype_translate(
                                            str(input_dtype)
                                        )
                                    ]
                                    * length
                                )
                            else:
                                raise RuntimeError(
                                    f"Missing tensor_meta for {gm_node.target}"
                                )
                        else:
                            node_dtype = self._torch_dtype_translate(
                                str(tensor_meta.dtype)
                            )
                            node_shape = tensor_meta.shape
                    elif num_returns > 1:
                        node_dtype = tuple(
                            [
                                self._torch_dtype_translate(str(val_item.dtype))
                                for val_item in val
                            ]
                        )
                        node_shape = tuple([val_item.shape for val_item in val])
                    else:
                        raise RuntimeError("Zero returns is not supported.")

                    oc_node = self._create_node(
                        str(gm_node.target.__name__),
                        gm_node.name,
                        gm_node.args,
                        node_users,
                        node_shape,
                        node_dtype,
                        node_kwargs=gm_node.kwargs,
                    )
                graph.add_node(oc_node)

            '''图变换与存储'''
            # transform_list = []
            self._imported_graphs.append(graph)
            self._imported_params[graph] = params_flat
            return _gm.forward
        
        return aot_module_simplified(
            gm,
            inputs,
            fw_compiler=_compiler,
            decompositions=self._aot_autograd_decomposition,
        )
    
    def __call__(self, gm: torch.fx.Graph, inputs: List[torch.Tensor]) -> Any:
        return self._compiler_fx(gm, inputs)

    def importer(self, model, *arg, **kwargs) -> List[Graph]:
        # if hasattr(model, "config") and model.config is not None:
        #     self._model_config = model.config.__class__.from_dict(
        #         model.config.to_dict()
        #     )

        model_opt = dynamo.optimize(self._compiler_fx)(model)
        model_opt(*arg, **kwargs)
        return self._imported_graphs

