from typing import Dict, List, Union, Sequence
import array

import mlir.ir as ir
from .utils import mlir_element_type_get

from mlir.dialects import (
    tosa,
    tensor,
)

from ..graph.operation import (
    AddOp,
    SliceOp,
)

def _normalize_binary_operator_shape(shp1, shp2):
    """Normalize the shape of two input tensors according to the broadcasting
    rule"""
    shp1 = list(shp1)
    shp2 = list(shp2)
    while len(shp1) < len(shp2):
        shp1.insert(0, 1)
    while len(shp2) < len(shp1):
        shp2.insert(0, 1)

    return shp1, shp2

def _create_mul_shift_operand() -> ir.Value:
    """Create the required shift operand for tosa.MulOp."""
    i8_type = ir.IntegerType.get_signless(8)
    tensor_type = ir.RankedTensorType.get([1], i8_type)
    zero_attr = ir.IntegerAttr.get(i8_type, 0)
    dense_attr = ir.DenseElementsAttr.get_splat(tensor_type, zero_attr)
    return tosa.ConstOp(dense_attr).results[0]

def _create_shape_operand(shape: Sequence[int]) -> ir.Value:
    """Create a tosa.shape value for reshape-like ops."""
    dims = [int(dim) for dim in shape]
    rank = len(dims)
    shape_type = ir.Type.parse(f"!tosa.shape<{rank}>")
    index_type = ir.IndexType.get()
    shape_attr = ir.DenseElementsAttr.get(
        array.array("q", dims),
        type=index_type,
        shape=[rank],
    )
    return tosa.ConstShapeOp(shape_type, shape_attr).result

def _scalar_to_tensor(
        scalar: Union[float, int], element_type: ir.Type, shape: List[int]
):
    '''标量转张量，MLIR所支持的数据类型'''
    if ir.FloatType.isinstance(element_type):
        element = ir.FloatAttr.get(element_type, float(scalar))
    else:
        element = ir.FloatAttr.get(element_type, int(scalar))
    attr = ir.DenseElementsAttr.get_splat(
        ir.RankedTensorType.get(shape, element_type), element
    )
    return tosa.ConstOp(attr).results[0]

def _normalize_binary_operator_args(arg1, arg2):
    '''统一输入类型'''
    if isinstance(arg1, ir.Value) and (
        isinstance(arg2, float) or isinstance(arg2, int)
    ):
        arg2 = _scalar_to_tensor(
            arg2,
            ir.RankedTensorType(arg1.type).element_type,
            ir.RankedTensorType(arg1.type).shape,
        )
        return arg1, arg2
    elif isinstance(arg2, ir.Value) and (
        isinstance(arg1, float) or isinstance(arg1, int)
    ):
        arg1 = _scalar_to_tensor(
            arg1,
            ir.RankedTensorType(arg2.type).element_type,
            ir.RankedTensorType(arg2.type).shape,
        )
        return arg1, arg2
    
    elif isinstance(arg1, ir.Value) and isinstance(arg2, ir.Value):
        return arg1, arg2
    elif (isinstance(arg1, float) or isinstance(arg1, int)) and (
        isinstance(arg2, float) or isinstance(arg2, int)
    ):
        if isinstance(arg1, float) or isinstance(arg2, float):
            arg1 = _scalar_to_tensor(arg1, ir.F32Type.get(), [1])
            arg2 = _scalar_to_tensor(arg2, ir.F32Type.get(), [1])
        else:
            arg1 = _scalar_to_tensor(arg1, ir.IntegerType.get_signless(32), [1])
            arg2 = _scalar_to_tensor(arg2, ir.IntegerType.get_signless(32), [1])
        return arg1, arg2
    else:
        raise ValueError(
            "Invalid input types %s and %s" % (type(arg1), type(arg2))
        )
    
def _gen_arith_binary_op(input1, input2, op_func):
    input1, input2 = _normalize_binary_operator_args(input1, input2)

    input1_shape = ir.RankedTensorType(input1.type).shape
    input2_shape = ir.RankedTensorType(input2.type).shape

    norm_input1_shape, norm_input2_shape = _normalize_binary_operator_shape(
        input1_shape, input2_shape
    )

    broadcasted_result_shp = []
    for dim1, dim2 in zip(norm_input1_shape, norm_input2_shape):
        broadcasted_result_shp.append(max(dim1, dim2))
    if input1_shape != norm_input1_shape:
        input1 = tosa.ReshapeOp(
            input1, _create_shape_operand(norm_input1_shape)
        ).result
    if input2_shape != norm_input2_shape:
        input2 = tosa.ReshapeOp(
            input2, _create_shape_operand(norm_input2_shape)
        ).result

    result_element_type = ir.RankedTensorType(input1.type).element_type
    result_tensor_type = ir.RankedTensorType.get(
        broadcasted_result_shp, result_element_type
    )
    # MulOp requires a shift parameter
    if op_func == tosa.MulOp:
        shift = _create_mul_shift_operand()
        op = op_func(result_tensor_type, input1, input2, shift)
    else:
        op = op_func(result_tensor_type, input1, input2)
    return op



def add_op(node: AddOp, symbol_table):
    """
    Import tensor addition operation.
    From buddy graph ir's `AddOp` operator to MLIR TOSA `add` operation.
    """
    input1 = symbol_table.get((str(node.args[0]), 0), node.args[0])
    input2 = symbol_table.get((str(node.args[1]), 0), node.args[1])
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    if isinstance(node.args[0], str) and isinstance(node.args[1], str):
        input1_dtype = ir.RankedTensorType(input1.type).element_type
        input2_dtype = ir.RankedTensorType(input2.type).element_type
        if input1_dtype != mlir_dtype:
            input1 = tosa.CastOp(
                ir.RankedTensorType.get(
                    ir.RankedTensorType(input1.type).shape,
                    mlir_dtype,
                ),
                input1,
            ).result
        if input2_dtype != mlir_dtype:
            input2 = tosa.CastOp(
                ir.RankedTensorType.get(
                    ir.RankedTensorType(input2.type).shape,
                    mlir_dtype,
                ),
                input2,
            ).result
    return _gen_arith_binary_op(input1, input2, tosa.AddOp)


def slice_op(node: SliceOp, symbol_table):
    """
    Import the slice operation.
    From buddy graph ir's `SliceOp` operator to MLIR TOSA `extract_slice`
    operation.
    """
    input_tensor = symbol_table.get((str(node.args[0]), 0))
    dim = node.args[1]
    start_idx = node.args[2]
    end_idx = node.args[3]

    sizes = ir.RankedTensorType(input_tensor.type).shape
    dtype = node.tensor_meta["dtype"]
    mlir_dtype = mlir_element_type_get(dtype)
    output_shape = list(node.tensor_meta["shape"])

    rank_diff = len(output_shape) - len(sizes)
    if rank_diff > 0:
        expanded_shape = [1] * rank_diff + list(sizes)
        shape_operand = _create_shape_operand(expanded_shape)
        input_tensor = tosa.ReshapeOp(input_tensor, shape_operand).result
        sizes = expanded_shape

    if start_idx < 0:
        start_idx += sizes[dim]

    if end_idx < 0:
        end_idx += sizes[dim]

    if start_idx < 0:
        start_idx = 0
    elif start_idx >= sizes[dim]:
        start_idx = sizes[dim]

    if end_idx < start_idx:
        end_idx = start_idx
    elif end_idx >= sizes[dim]:
        end_idx = sizes[dim]

    new_sizes = [x for x in sizes]
    new_sizes[dim] = end_idx - start_idx
    new_sizes_attr = ir._denseI64ArrayAttr(new_sizes, None)

    offsets = [0] * len(sizes)
    offsets[dim] = start_idx
    offsets_attr = ir._denseI64ArrayAttr(offsets, None)

    strides = [1] * len(sizes)
    strides_attr = ir._denseI64ArrayAttr(strides, None)

    extract_slice_result_type = ir.RankedTensorType.get(new_sizes, mlir_dtype)
    if new_sizes == sizes:
        return input_tensor
    op = tensor.ExtractSliceOp(
        extract_slice_result_type,
        input_tensor,
        [],
        [],
        [],
        offsets_attr,
        new_sizes_attr,
        strides_attr,
    )

    return op


ops_registry = {
    "AddOp": add_op,
    "SliceOp":slice_op,
}

