from numba import njit

import minitorch
import minitorch.fast_ops

# MAP
print("MAP")
tmap = minitorch.fast_ops.tensor_map(njit()(minitorch.operators.id))
out, a = minitorch.zeros((10,)), minitorch.zeros((10,))
tmap(*out.tuple(), *a.tuple())
tmap.parallel_diagnostics(level=3)

# ZIP
print("\nZIP")
out, a, b = minitorch.zeros((10,)), minitorch.zeros((10,)), minitorch.zeros((10,))
tzip = minitorch.fast_ops.tensor_zip(njit()(minitorch.operators.eq))

tzip(*out.tuple(), *a.tuple(), *b.tuple())
tzip.parallel_diagnostics(level=3)

# REDUCE
print("\nREDUCE")
out, a = minitorch.zeros((1,)), minitorch.zeros((10,))
treduce = minitorch.fast_ops.tensor_reduce(njit()(minitorch.operators.add))

treduce(*out.tuple(), *a.tuple(), 0)
treduce.parallel_diagnostics(level=3)


# MM
print("\nMATRIX MULTIPLY")
out, a, b = (
    minitorch.zeros((1, 10, 10)),
    minitorch.zeros((1, 10, 20)),
    minitorch.zeros((1, 20, 10)),
)
tmm = minitorch.fast_ops.tensor_matrix_multiply

tmm(*out.tuple(), *a.tuple(), *b.tuple())
tmm.parallel_diagnostics(level=3)
