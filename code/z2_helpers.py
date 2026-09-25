from . import bitgauss_wrappers as z2lin
# if for some reason bitgauss installation doesn't work, it can be changed with galois using the line below
# import galois_wrappers as z2lin
import numpy as np

def stagger_kernel(A, dim1):
    """
    Computes the a Z2 kernel isomorphism K of the matrix A, whose rows are divided into 3 blocks such that K is a 3 x 3 block matrix that is block upper triangular and every diagonal block is injective
    """
    A_rref = z2lin.rref(A)
    pivs = get_pivots(A_rref)
    non_pivs = sorted(set(range(0,A.shape[1])) - set(pivs))
    kerdim = len(non_pivs) # dimension of kernel is number of non-pivots
    ker = np.zeros((A.shape[1], kerdim), dtype=int)
    ker[non_pivs, :] = np.eye(kerdim, dtype=int)
    ker[pivs, :] = A_rref[:len(pivs),:][:, non_pivs]
    cum_dim1 = np.r_[0, np.cumsum(dim1)]
    ker_dims = (np.searchsorted(non_pivs, cum_dim1[1:], side="left") - np.searchsorted(non_pivs, cum_dim1[:-1], side="left")).tolist()
    #ker_dims = [bisect_left(A, cum_dim1[i + 1]) - bisect_left(A, cum_dim1[i]) for i in range(len(dim1))]
    return ker, ker_dims


# extract list of pivot columns from a matrix in RREF (row-reduced echolon form)
def get_pivots(A, return_non_pivs = False):
    zero_rows = ~A.any(axis=1)
    nr_pivots = zero_rows.argmax() if zero_rows.any() else len(zero_rows)
    if A.shape[1] == 0:
        # what is this if clause for??? can skip??
        pivs = np.zeros((nr_pivots,), dtype=int).tolist()
    else:
        # argmax implementation is bad so we have to byhand implement an exception
        pivs = A[:nr_pivots,:].argmax(axis=1).tolist()
    
    if return_non_pivs:
        non_pivs = sorted(set(range(0,A.shape[1])) - set(pivs))
        return pivs, non_pivs
    else:
        return pivs
    
# returns the RREF with zero rows removed, the pivots, and the non-pivots
def rref_trim_pivs(A):
    A_rref_full = z2lin.rref(A)
    zero_rows = ~A_rref_full.any(axis=1)
    nr_pivots = zero_rows.argmax() if zero_rows.any() else len(zero_rows)
    A_rref = A_rref_full[:nr_pivots, :]
    #print("A_rref\n", A_rref.shape)
    if A_rref.shape[0]==0:
        # extra case due to bad implementation of argmax
        pivs = []
    else:
        pivs = A_rref.argmax(axis=1).tolist()
    non_pivs = sorted(set(range(0,A.shape[1])) - set(pivs))
    return A_rref, pivs, non_pivs

# returns: (1) list of column numbers restricted to which A is injective, (2) minimal list of computational-basis vectors that complete columns to a basis of both A and B
def remove_image(A, B):
    AB = np.hstack([A, B])
    AB_red = z2lin.rref(AB)
    pivots = get_pivots(AB_red)
    n = A.shape[1]
    A_pivots = [j for j in pivots if j < n]
    B_pivots = [j - n for j in pivots if j >= n]
    return A_pivots, B_pivots

# returns a matrix B such that the stack (A,B) is injective
def image_completion(A):
    _, compl_pivs = remove_image(A, np.eye(A.shape[0]))
    return np.eye(A.shape[0],dtype=int)[:, compl_pivs]

# returns: (1) list of column numbers restricted to which A is injective
def injectify(A):
    A_red = z2lin.rref(A)
    return get_pivots(A_red)

def z2_injectify(A):
    A_red = z2lin.rref(A)
    return np.eye(A.shape[1],dtype=int)[:, get_pivots(A_red)]

# returns an injective version of A and the quotient as well
def z2_injquot(A):
    A_rref = z2lin.rref(A)
    pivs = get_pivots(A_rref)
    non_pivs = sorted(set(range(0,A.shape[1])) - set(pivs))
    nonpiv_to_piv = A_rref[:len(pivs), non_pivs]
    A_inj = A[:, pivs]
    A_quot = np.zeros((len(pivs), A.shape[1]), dtype=int)
    A_quot[range(len(pivs)), pivs] = 1
    A_quot[:, non_pivs] = nonpiv_to_piv
    return A_inj, A_quot

# As: list of matrices with the same 0th dimension
def z2_multi_injectify(As):
    Astack = np.hstack(As)
    Astack_red = z2lin.rref(Astack)
    pivots = get_pivots(Astack_red)
    res = []
    tot_size = 0
    for A in As:
        l = A.shape[1]
        piv_list = [j - tot_size for j in pivots if j >= tot_size and j< tot_size+l]
        res.append(np.eye(l,dtype=int)[:, piv_list])
        tot_size += l
    return res

def rref_with_transform(A):
    A_id = np.hstack([A, np.eye(A.shape[0], dtype=int)])
    A_id_rref = z2lin.rref(A_id)
    A_rref = A_id_rref[:, :A.shape[1]]
    A_transform = A_id_rref[:, A.shape[1]:]
    return A_rref, A_transform

def get_solve_helper(A):
    Arref, Atrans = rref_with_transform(A)
    piv = get_pivots(Arref)
    return A.shape[1], piv, Atrans

# Solves Ax=b for x. pivs and transform are the pivot columns and the transform for the RREF of A
def solve_with_helper(size, pivs, transform, b):
    trans_b = transform @ b
    if not np.all(trans_b[len(pivs):] % 2 == 0):
        raise ValueError("Linear equation has no solution")
    else:
        sol = np.zeros((size,),dtype=int)
        sol[pivs] = trans_b[:len(pivs)] % 2
        return sol
