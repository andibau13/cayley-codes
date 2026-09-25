from bitgauss import BitMatrix
import numpy as np

def numpy_to_bitgauss(A):
    return BitMatrix.from_numpy_bool(A==1)
def bitgauss_to_numpy(A):
    return A.to_numpy_bool().astype(int)

def kernel(A):
    if A.shape[0]==0:
        return np.eye(A.shape[1], dtype=int)
    bm = BitMatrix.from_numpy_bool(A==1)
    return bm.nullspace_matrix().to_numpy_bool().astype(int)
def rref(A):
    bm = BitMatrix.from_numpy_bool(A==1)
    bm.gauss(full=True)
    return bm.to_numpy_bool().astype(int)

