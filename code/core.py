"""
Maps between free modules over the group ring R = F_2[F_2] (F_2 = free group on a,b).

Conventions
-----------
Group elements of F_2 are reduced words, stored as tuples of nonzero ints:
    1 -> a,  -1 -> a^-1,  2 -> b,  -2 -> b^-1.
A map f: R^n -> R^m is an m x n matrix over R acting by left multiplication on
column vectors, so R^n is a *right* R-module and composition is the matrix
product f @ g.  Right multiplication by a group element w ("translation") is the
only side that commutes with f.

Coefficients are stored as a single (s*m) x n binary matrix together with one
overall support (list of s group elements), i.e. the same support is used for
every entry of the matrix.  Row (i, k) of the coefficient matrix (i = output
coordinate, k = support index) sits at row index i*s + k, so that
mat.reshape(m, s, n) gives the coefficient tensor.

Vectors/columns of a free module R^n restricted to a finite set of group
elements ("support") are always flattened in the same layout:
    index = coordinate * len(support) + support_index.
"""

import numpy as np
import z2_helpers
from z2_helpers import z2lin

GENS = (1, -1, 2, -2)
E = ()  # identity word


# ---------------------------------------------------------------- free group F_2

def w_mul(u, v):
    """Product of two reduced words."""
    i = 0
    while i < len(u) and i < len(v) and u[len(u) - 1 - i] == -v[i]:
        i += 1
    return u[: len(u) - i] + v[i:]


def w_inv(u):
    return tuple(-x for x in reversed(u))


def w_str(u):
    return "e" if not u else "".join("aAbB"[(2 * (abs(x) - 1)) + (x < 0)] for x in u)


def star(l):
    """Depth-l star: all reduced words of length <= l, in BFS order."""
    words, frontier = [E], [E]
    for _ in range(l):
        frontier = [w + (x,) for w in frontier for x in GENS if not w or w[-1] != -x]
        words += frontier
    return words


def _sort_words(words):
    return sorted(words, key=lambda w: (len(w), w))


# -------------------------------------------------------------------- the map class

class ModuleMap:
    """A map f: R^n -> R^m over R = F_2[F_2]."""

    def __init__(self, support, mat, m=None):
        self.support = list(support)
        self.s = len(self.support)
        self.index = {w: k for k, w in enumerate(self.support)}
        self.mat = np.asarray(mat, dtype=int) % 2
        self.n = self.mat.shape[1]
        self.m = self.mat.shape[0] // self.s if m is None else m
        assert self.mat.shape[0] == self.m * self.s

    # -- basic access

    @property
    def blocks(self):
        """Coefficient tensor of shape (m, s, n)."""
        return self.mat.reshape(self.m, self.s, self.n)

    def entry(self, i, j):
        """The ring element f_ij, as a list of words."""
        return [self.support[k] for k in np.nonzero(self.blocks[i, :, j])[0]]

    def column(self, j):
        """The j-th column as a map R^1 -> R^m, with trimmed support."""
        return ModuleMap(self.support, self.blocks[:, :, j: j + 1].reshape(self.m * self.s, 1),
                         self.m).trimmed()

    def is_zero(self):
        return not self.mat.any()

    def weights(self):
        """Number of nonzero (coordinate, group element) entries of each column."""
        return self.mat.sum(axis=0)

    def depths(self):
        """Largest word length occurring in each column (support radius)."""
        lens = np.array([len(w) for w in self.support])
        return np.array([lens[self.blocks[:, :, j].any(axis=0)].max(initial=0) for j in range(self.n)])

    @classmethod
    def from_entries(cls, m, n, entries):
        """entries: dict (i, j) -> iterable of words (repeats cancel mod 2)."""
        support = _sort_words({w for ws in entries.values() for w in ws}) or [E]
        mat = np.zeros((m * len(support), n), dtype=int)
        blocks = mat.reshape(m, len(support), n)
        idx = {w: k for k, w in enumerate(support)}
        for (i, j), ws in entries.items():
            for w in ws:
                blocks[i, idx[w], j] ^= 1
        return cls(support, mat, m)

    def trimmed(self):
        """Drop support elements with only zero coefficients."""
        keep = np.nonzero(self.blocks.any(axis=(0, 2)))[0]
        if len(keep) == 0:
            keep = np.array([0])  # keep the support non-empty
        support = [self.support[k] for k in keep]
        return ModuleMap(support, self.blocks[:, keep, :].reshape(self.m * len(keep), self.n), self.m)

    def shifted(self, w):
        """Right translation f -> f*w (relabels the support, same coefficients)."""
        return ModuleMap([w_mul(u, w) for u in self.support], self.mat, self.m)

    def __repr__(self):
        rows = [
            "  [" + ", ".join("+".join(map(w_str, self.entry(i, j))) or "0" for j in range(self.n)) + "]"
            for i in range(self.m)
        ]
        return f"ModuleMap R^{self.n} -> R^{self.m} (|supp|={self.s})\n" + "\n".join(rows)

    # -- expansion to F_2 matrices

    def dense_columns(self, support):
        """
        The columns of self written in the flat (m, support) layout: (m*|support|, n).
        `support` is a list of words or an already built word -> index dict.
        """
        if isinstance(support, dict):
            sup_idx = support
        else:
            sup_idx = {w: k for k, w in enumerate(support)}
        idx = np.array([sup_idx[w] for w in self.support], dtype=int)
        S = len(sup_idx)
        out = np.zeros((self.m * S, self.n), dtype=int)
        out.reshape(self.m, S, self.n)[:, idx, :] = self.blocks
        return out

    def expand(self, support_in):
        """
        F_2 matrix of f restricted to inputs supported on support_in.
        Returns (M, support_out) with M of shape (m*|support_out|, n*|support_in|),
        support_out = supp(f) * support_in.
        """
        prods = [[w_mul(u, g) for g in support_in] for u in self.support]
        support_out = _sort_words({h for row in prods for h in row})
        out_idx = {w: k for k, w in enumerate(support_out)}
        N, So = len(support_in), len(support_out)
        M = np.zeros((self.m * So, self.n * N), dtype=int)
        Mv = M.reshape(self.m, So, self.n, N)
        cols = np.arange(N)
        blocks = self.blocks
        for k in range(self.s):
            rows_k = np.array([out_idx[h] for h in prods[k]], dtype=int)
            for i, j in zip(*np.nonzero(blocks[:, k, :])):
                Mv[i, rows_k, j, cols] = 1
        return M, support_out


def compose(f, g):
    """f: R^n -> R^m, g: R^p -> R^n  ==>  f*g: R^p -> R^m."""
    assert f.n == g.m
    support = _sort_words({w_mul(u, v) for u in f.support for v in g.support})
    idx = {w: k for k, w in enumerate(support)}
    H = np.zeros((f.m, len(support), g.n), dtype=int)
    for ku, u in enumerate(f.support):
        for kv, v in enumerate(g.support):
            H[:, idx[w_mul(u, v)], :] ^= (f.blocks[:, ku, :] @ g.blocks[:, kv, :]) % 2
    return ModuleMap(support, H.reshape(f.m * len(support), g.n), f.m).trimmed()


def hermitian_transpose(f):
    """
    f^dagger: R^m -> R^n with (f^dagger)_ji = (f_ij)^*, where * is the antipode
    g -> g^-1 of R (extended linearly).  This is the adjoint of f for the pairing
    <x, y> = sum_i (x_i)^* y_i, and it is an anti-involution:
    (f^dagger)^dagger = f and (f g)^dagger = g^dagger f^dagger.
    """
    return ModuleMap([w_inv(u) for u in f.support],
                     f.blocks.transpose(2, 1, 0).reshape(f.n * f.s, f.m), f.n)


# ------------------------------------------------------------------------ syzygies

def _valid_shifts(supp, star_set):
    """All w with supp*w contained in star_set (candidates come from u^-1 * star)."""
    uinv = w_inv(supp[0])
    cand = {w_mul(uinv, h) for h in star_set}
    return [w for w in cand if all(w_mul(t, w) in star_set for t in supp)]


def syzygies(f, l_max, verbose=False):
    """
    Independent generators g: R^k -> R^n of the syzygies of f: R^n -> R^m, i.e.
    f*g = 0 and the image of g contains every kernel element of f supported on
    the depth-l_max star.

    Computed star by star: at depth l we take the kernel of f restricted to the
    depth-l star and quotient by all translates (of already found generators)
    that still fit inside the depth-l star.  The list of numbers of new
    generators per depth is stored on the result as `g.level_counts`.
    """
    gens = []          # accepted generators, each a ModuleMap R^1 -> R^n
    counts = []
    for l in range(l_max + 1):
        Sl = star(l)
        Sl_set = set(Sl)
        Sl_idx = {w: k for k, w in enumerate(Sl)}
        K = z2lin.kernel(f.expand(Sl)[0])

        # span of all translates of known generators that fit in the depth-l star
        prev = [x.shifted(w).dense_columns(Sl_idx)
                for x in gens for w in _valid_shifts(x.support, Sl_set)]
        prev = np.hstack(prev) if prev else np.zeros((f.n * len(Sl), 0), dtype=int)

        n_new = 0
        while K.shape[1]:
            new_cols = z2_helpers.remove_image(prev, K)[1]
            if not new_cols:
                break
            j = new_cols[0]
            x = ModuleMap(Sl, K[:, j: j + 1], f.n).trimmed()
            gens.append(x)
            n_new += 1
            prev = np.hstack([prev] + [x.shifted(w).dense_columns(Sl_idx)
                                       for w in _valid_shifts(x.support, Sl_set)])
            K = K[:, j + 1:]
        counts.append(n_new)
        if verbose:
            print(f"l={l}: {n_new} new generator(s)")

    S = star(l_max)
    S_idx = {w: k for k, w in enumerate(S)}
    cols = [x.dense_columns(S_idx) for x in gens]
    cols = np.hstack(cols) if cols else np.zeros((f.n * len(S), 0), dtype=int)
    g = ModuleMap(S, cols, f.n).trimmed()
    g.level_counts = counts
    return g


def star_span(g, l):
    """
    F_2 matrix (in the flat (m, star(l)) layout) whose columns are all translates of
    the columns of g that still fit inside the depth-l star, i.e. a basis-free
    description of im(g) restricted to that star.
    """
    Sl = star(l)
    Sl_idx = {w: k for k, w in enumerate(Sl)}
    Sl_set = set(Sl)
    cols = [x.shifted(w).dense_columns(Sl_idx)
            for x in map(g.column, range(g.n)) for w in _valid_shifts(x.support, Sl_set)]
    return np.hstack(cols) if cols else np.zeros((g.m * len(Sl), 0), dtype=int)


def module_eq_on_star(g, h, l):
    """Do im(g) and im(h) agree on the depth-l star?  (as F_2 spans of translates)"""
    A, B = star_span(g, l), star_span(h, l)
    return not z2_helpers.remove_image(A, B)[1] and not z2_helpers.remove_image(B, A)[1]


# ------------------------------------------------------------------- infinite codes

def random_gens(n, m, l, w, rng=None):
    """m random elements of R^n of weight w supported on the depth-l star, as a map R^m -> R^n."""
    rng = np.random.default_rng() if rng is None else rng
    S = star(l)
    mat = np.zeros((n * len(S), m), dtype=int)
    for j in range(m):
        mat[rng.choice(n * len(S), size=w, replace=False), j] = 1
    return ModuleMap(S, mat, n).trimmed()


def generate_infinite_code(n, m, l_max, l_init, w_init, rng=None, f=None, verbose=False):
    """
    Random translation-invariant CSS code on the Cayley graph of F_2 with n qubits
    per vertex (see cayley_codes.md).

    m < n random Z-type generators of weight w_init supported on the depth-l_init star
    are assembled into f: R^m -> R^n (columns = the generators).  Then

        H_X = ker(f^dagger),   H_Z = ker(H_X^dagger),

    with ker = `syzygies(., l_max)`.  H_X collects the X operators commuting with all
    translates of the initial Z operators (x commutes with all translates of y iff
    y^dagger x = 0), and H_Z then collects *all* Z operators commuting with those,
    which generally contains more than the initial f.  Both are returned as maps into
    R^n (columns = stabilizer generators).  f can be passed in explicitly instead of
    being sampled.
    """
    if f is None:
        f = random_gens(n, m, l_init, w_init, rng)
    H_X = syzygies(hermitian_transpose(f), l_max)
    H_Z = syzygies(hermitian_transpose(H_X), l_max)
    if verbose:
        print(f"f:   {f.n} Z gen(s), weights {f.weights()}, depths {f.depths()}")
        print(f"H_X: {H_X.n} gen(s) per vertex, level counts {H_X.level_counts}, "
              f"weights {H_X.weights()}, depths {H_X.depths()}")
        print(f"H_Z: {H_Z.n} gen(s) per vertex, level counts {H_Z.level_counts}, "
              f"weights {H_Z.weights()}, depths {H_Z.depths()}")
    return H_X, H_Z
