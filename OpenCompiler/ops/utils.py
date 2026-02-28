from typing import Dict
import mlir.ir as ir 

from ..graph.type import TensorDType

def mlir_element_type_get(type_name):

    match type_name:
        case TensorDType.Float32:
            return ir.F32Type.get()
        case TensorDType.Float64:
            return ir.F64Type.get()
        case TensorDType.Int8:
            return ir.IntegerType.get_signless(8)

