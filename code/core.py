"""
Maps between free modules over the group ring R = F_2[F_2] (F_2 = free group on a,b).

Conventions
-----------
Group elements of F_2 are written as reduced words, tuples of nonzero ints:
    1 -> a,  -1 -> a^-1,  2 -> b,  -2 -> b^-1.
Internally they are *integers*: the index of the element in a BFS enumeration of
the Cayley graph (`BALL`, grown lazily).  Index 0 is the identity, and the radius-l
ball around it is exactly the index range [0, BALL.size(l)).

Sizes of finite windows are given by a *diameter* `diam`: `ball(diam)` is the ball
around the identity for even diam, and the ball around the midpoint of an edge
{e, a} or {e, b} for odd diam (see `ball`).

A map f: R^n -> R^m is an m x n matrix over R acting by left multiplication on
column vectors, so R^n is a *right* R-module and composition is the matrix
product f @ g.  Right multiplication by a group element w ("translation") is the
only side that commutes with f.

Coefficients are stored as a single (s*m) x n binary matrix together with one
overall support (sorted int array of s group elements), i.e. the same support is
used for every entry of the matrix.  Row (i, k) of the coefficient matrix
(i = output coordinate, k = support index) sits at row index i*s + k, so that
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
# dtype of all F_2 matrices: 8x less memory than int64, and parity survives the
# mod-256 wraparound of uint8 sums/products, so `(A @ B) % 2` stays correct
BIT = np.uint8


# ---------------------------------------------------------------- free group F_2 (words)

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


# ------------------------------------------------------------ integer-indexed ball

class Ball:
    """
    The ball of the Cayley graph around the identity, enumerated by BFS and grown on
    demand.  Elements are ints (BFS index); BFS order is stable under growth, so indices
    never change.  Only `mul` / `inv` / `gens` on normal forms are used to build it.

    Arrays (all indexed by element):
        length[g]       word length
        parent[g], letter[g]   g = parent[g] * gens[letter[g]] (BFS tree; -1 for e)
        R[k, g]         g * gens[k], or -1 if outside the current ball; R[:, -1] = -1
                        is a sentinel column so that -1 propagates under fancy indexing
        inv[g]          g^-1
    """

    def __init__(self, identity=E, gens=tuple((x,) for x in GENS), mul=w_mul, inv=w_inv):
        self.gens, self._mul, self._inv = gens, mul, inv
        self.elems, self.index = [identity], {identity: 0}
        self.sizes = [1]
        self.parent, self.letter = np.array([-1]), np.array([-1])
        self.length, self.inv = np.array([0]), np.array([0])
        self.R = np.full((len(gens), 2), -1)

    @property
    def radius(self):
        return len(self.sizes) - 1

    def size(self, l):
        self.grow(l)
        return self.sizes[l]

    def grow(self, l):
        if l <= self.radius:
            return
        old_N = len(self.elems)
        par, let = [], []
        while self.radius < l:
            lo, hi = (self.sizes[-2] if self.radius else 0), self.sizes[-1]
            for i in range(lo, hi):
                for k, x in enumerate(self.gens):
                    h = self._mul(self.elems[i], x)
                    if h not in self.index:
                        self.index[h] = len(self.elems)
                        self.elems.append(h)
                        par.append(i)
                        let.append(k)
            self.sizes.append(len(self.elems))
        N = len(self.elems)
        self.parent = np.concatenate([self.parent, par]).astype(int)
        self.letter = np.concatenate([self.letter, let]).astype(int)
        self.length = np.repeat(np.arange(len(self.sizes)), np.diff([0] + self.sizes))
        self.inv = np.concatenate([self.inv, [self.index[self._inv(self.elems[g])]
                                              for g in range(old_N, N)]]).astype(int)
        R = np.full((len(self.gens), N + 1), -1)
        R[:, :old_N] = self.R[:, :old_N]
        # only the old outer sphere and the new elements can have neighbours outside
        todo = np.nonzero((R[:, :N] < 0).any(axis=0))[0]
        for g in todo:
            for k, x in enumerate(self.gens):
                R[k, g] = self.index.get(self._mul(self.elems[g], x), -1)
        self.R = R

    def elem(self, word):
        """Integer element of a (not necessarily reduced) word in the letters GENS."""
        self.grow(len(word))
        g = 0
        for x in word:
            g = self.R[GENS.index(x), g]
        return int(g)

    def word(self, g):
        return self.elems[g]

    def mul(self, A, B):
        """
        Product table T[i, j] = A[i] * B[j] (int arrays of elements), -1 where the product
        lies outside the current ball.  Computed along the BFS tree of B: if b = p * x then
        a*b = (a*p)*x, one fancy-indexing step per BFS level.
        """
        A, B = np.atleast_1d(A), np.atleast_1d(B)
        lB = self.length[B].max(initial=0)
        T = np.empty((len(A), self.sizes[lB]), dtype=int)
        T[:, 0] = A
        for d in range(1, lB + 1):
            lvl = np.arange(self.sizes[d - 1], self.sizes[d])
            T[:, lvl] = self.R[self.letter[lvl], T[:, self.parent[lvl]]]
        return T[:, B]


BALL = Ball()


def _star(l):
    """Radius-l ball around the identity, as the int array of its elements (= range, by BFS order)."""
    return np.arange(BALL.size(l))


EDGE_GENS = ((1,), (2,))  # centre edges {e, a}, {e, b} of the odd-diameter balls
_BALLS = {}


def ball(diam, s=0):
    """
    The ball of diameter `diam`, as a sorted int array:
      even diam: all g at distance <= diam/2 from the identity,
      odd diam:  all g at distance <= diam//2 from e or from x = EDGE_GENS[s], i.e. the
                 ball of radius diam/2 around the midpoint of the edge {e, x}.
    The distance is d(g, h) = |g h^-1| (invariant under translation = right
    multiplication), so the second half is _star(diam//2) * x.  `s` is ignored for even
    diam.  Every finite subset of the tree of diameter D lies in exactly one translate
    of exactly one of the diameter-D balls (its centre is a unique vertex or edge).
    """
    s = s if diam % 2 else 0
    if (diam, s) not in _BALLS:
        S = _star(diam // 2)
        if diam % 2:
            BALL.grow(diam // 2 + 1)
            S = np.union1d(S, BALL.mul(S, BALL.elem(EDGE_GENS[s]))[:, 0])
        _BALLS[diam, s] = S
    return _BALLS[diam, s]


def ball_types(diam):
    """The values of s giving the different diameter-diam balls (2 for odd diam, else 1)."""
    return range(len(EDGE_GENS) if diam % 2 else 1)


def _elem(w):
    """Accept a group element either as int or as word."""
    return BALL.elem(w) if isinstance(w, tuple) else int(w)


def _maxlen(support):
    return BALL.length[support].max(initial=0)


# -------------------------------------------------------------------- the map class

class ModuleMap:
    """A map f: R^n -> R^m over R = F_2[F_2]."""

    def __init__(self, support, mat, m=None):
        support = np.asarray(support, dtype=int)
        mat = (np.asarray(mat) % 2).astype(BIT)
        s, n = len(support), mat.shape[1]
        m = mat.shape[0] // s if m is None else m
        assert mat.shape[0] == m * s
        order = np.argsort(support, kind="stable")
        self.support = support[order]
        self.mat = mat.reshape(m, s, n)[:, order, :].reshape(m * s, n)
        self.s, self.m, self.n = s, m, n

    # -- basic access

    @property
    def blocks(self):
        """Coefficient tensor of shape (m, s, n)."""
        return self.mat.reshape(self.m, self.s, self.n)

    def entry(self, i, j):
        """The ring element f_ij, as a list of words."""
        return [BALL.word(g) for g in self.support[self.blocks[i, :, j] == 1]]

    def column(self, j):
        """The j-th column as a map R^1 -> R^m, with trimmed support."""
        return ModuleMap(self.support, self.blocks[:, :, j: j + 1].reshape(self.m * self.s, 1),
                         self.m).trimmed()

    def is_zero(self):
        return not self.mat.any()

    def weights(self):
        """Number of nonzero (coordinate, group element) entries of each column."""
        return self.mat.sum(axis=0)

    def diams(self):
        """Diameter of the support of each column (largest distance |u v^-1| within it)."""
        BALL.grow(2 * _maxlen(self.support))
        D = BALL.length[BALL.mul(self.support, BALL.inv[self.support])]  # (s, s)
        return np.array([D[np.ix_(o, o)].max(initial=0) for o in self.blocks.any(axis=0).T], dtype=int)

    @classmethod
    def from_entries(cls, m, n, entries):
        """entries: dict (i, j) -> iterable of words or ints (repeats cancel mod 2)."""
        coef = {}
        for (i, j), ws in entries.items():
            for w in ws:
                key = (i, _elem(w), j)
                coef[key] = coef.get(key, 0) ^ 1
        support = np.unique([g for (_, g, _) in coef] or [0])
        mat = np.zeros((m, len(support), n), dtype=BIT)
        for (i, g, j), c in coef.items():
            mat[i, np.searchsorted(support, g), j] = c
        return cls(support, mat.reshape(m * len(support), n), m)

    def trimmed(self):
        """Drop support elements with only zero coefficients."""
        keep = np.nonzero(self.blocks.any(axis=(0, 2)))[0]
        if len(keep) == 0:
            keep = np.array([0])  # keep the support non-empty
        return ModuleMap(self.support[keep], self.blocks[:, keep, :].reshape(self.m * len(keep), self.n),
                         self.m)

    def shifted(self, w):
        """Right translation f -> f*w (relabels the support, same coefficients)."""
        w = _elem(w)
        BALL.grow(_maxlen(self.support) + BALL.length[w])
        return ModuleMap(BALL.mul(self.support, w)[:, 0], self.mat, self.m)

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
        `support` is a sorted int array containing self.support.
        """
        S = len(support)
        out = np.zeros((self.m, S, self.n), dtype=BIT)
        out[:, np.searchsorted(support, self.support), :] = self.blocks
        return out.reshape(self.m * S, self.n)

    def compactify(self, comp):
        """
        Push f through a quotient F_2 -> G (a `Compactification`), giving the ordinary
        binary matrix of shape (m*|G|, n*|G|) in the flat layout
        index = coordinate * |G| + group element index.
        """
        N = comp.N
        out = np.zeros((self.m * N, self.n * N), dtype=BIT)
        V = out.reshape(self.m, N, self.n, N)
        cols = np.arange(N)[None, :]
        for k, u in enumerate(self.support):
            rows = comp.left_perm(BALL.word(u))[None, :]  # rows[g] = index of pi(u) * g
            i, j = np.nonzero(self.blocks[:, k, :])
            # distinct (i, j) hit distinct entries; ^= because distinct words of F_2
            # can collapse onto the same element of G
            V[i[:, None], rows, j[:, None], cols] ^= 1
        return out

    def expand(self, support_in):
        """
        F_2 matrix of f restricted to inputs supported on support_in (int array).
        Returns (M, support_out) with M of shape (m*|support_out|, n*|support_in|),
        support_out = supp(f) * support_in (sorted).
        """
        support_in = np.asarray(support_in, dtype=int)
        BALL.grow(_maxlen(self.support) + _maxlen(support_in))
        T = BALL.mul(self.support, support_in)                      # (s, N)
        support_out, rows = np.unique(T, return_inverse=True)
        rows = rows.reshape(T.shape)
        N, So = len(support_in), len(support_out)
        M = np.zeros((self.m, So, self.n, N), dtype=BIT)
        i, k, j = np.nonzero(self.blocks)
        M[i[:, None], rows[k], j[:, None], np.arange(N)[None, :]] = 1  # u*g distinct for distinct u
        return M.reshape(self.m * So, self.n * N), support_out


def compose(f, g):
    """f: R^n -> R^m, g: R^p -> R^n  ==>  f*g: R^p -> R^m."""
    assert f.n == g.m
    BALL.grow(_maxlen(f.support) + _maxlen(g.support))
    support, idx = np.unique(BALL.mul(f.support, g.support), return_inverse=True)
    # P[i, u, v, j] = sum_k f[i, u, k] g[k, v, j]  (coefficient of the product u*v)
    P = (f.blocks.reshape(f.m * f.s, f.n) @ g.blocks.reshape(g.m, g.s * g.n))
    H = np.zeros((f.m, len(support), g.n), dtype=BIT)
    np.add.at(H, (slice(None), idx.ravel()), P.reshape(f.m, f.s * g.s, g.n))
    return ModuleMap(support, H.reshape(f.m * len(support), g.n) % 2, f.m).trimmed()


def hermitian_transpose(f):
    """
    f^dagger: R^m -> R^n with (f^dagger)_ji = (f_ij)^*, where * is the antipode
    g -> g^-1 of R (extended linearly).  This is the adjoint of f for the pairing
    <x, y> = sum_i (x_i)^* y_i, and it is an anti-involution:
    (f^dagger)^dagger = f and (f g)^dagger = g^dagger f^dagger.
    """
    return ModuleMap(BALL.inv[f.support], f.blocks.transpose(2, 1, 0).reshape(f.n * f.s, f.m), f.n)


# ------------------------------------------------------------------------ syzygies

def translates(x, window):
    """
    All translates x*w of the single column x (a map R^1 -> R^m) whose support fits
    inside `window` (a sorted int array, e.g. `ball(diam, s)`), as an F_2 matrix
    (m*|window|, #translates) in the flat layout.  Candidates w = u0^-1 * h (u0 in
    supp(x), h in the window), then filtered.
    """
    window = np.asarray(window)
    N = len(window)
    BALL.grow(_maxlen(window) + _maxlen(x.support))
    cand = BALL.mul(BALL.inv[x.support[:1]], window)[0]
    T = BALL.mul(x.support, cand)                     # (s, #cand), -1 outside the ball
    pos = np.minimum(np.searchsorted(window, T), N - 1)
    P = pos[:, (window[pos] == T).all(axis=0)]        # positions inside the window
    out = np.zeros((x.m * N, P.shape[1]), dtype=BIT)
    i, k = np.nonzero(x.blocks[:, :, 0])
    out[i[:, None] * N + P[k], np.arange(P.shape[1])[None, :]] = 1
    return out


def _hstack(cols, rows):
    return np.hstack(cols) if cols else np.zeros((rows, 0), dtype=BIT)


def syzygies(f, max_diam, margin_rad=0, same_diam_pruning=True, verbose=False):
    """
    Generators g: R^k -> R^n of the syzygies of f: R^n -> R^m, i.e. f*g = 0 and the
    R-module generated by the columns of g contains the kernel of f restricted to the
    diameter-max_diam balls (both of them if max_diam is odd).

    For diam = 0, 1, ..., max_diam (half-integer radius steps) and each diameter-diam
    ball W, let W+ = ball(diam + 2*margin_rad) be the ball with the same centre and
    margin_rad larger radius:
    (1) Candidates: an F_2 basis of the kernel of f restricted to W, minus the span of
        the translates inside W+ of the generators of smaller diameter.  Each dropped
        vector is an exact R-combination of those generators.  On the tree, a new
        candidate has support diameter exactly diam with the centre of W, so its only
        translate fitting in W is itself.
    (2) Same-diameter pruning (if same_diam_pruning): a new candidate x is dropped if it
        lies in the span of the translates, inside x's W+, of the generators of smaller
        diameter and the remaining other diameter-diam candidates (tested last found
        first).  For F_2 with margin_rad = 0 this never removes anything (see (1)), but
        for groups where a ball can contain several translates of a candidate it can.
    Generators of smaller diameter are never removed.  A vector dropped at diameter d'
    never reappears later: its translate x*w fitting in a larger ball W lies in a
    concentric ball inside W, so its witnesses lie inside the W+ of W.

    The generators come in order of increasing diameter.  Attributes of g:
    `level_counts[d]` = number of generators of diameter d, `n_pruned` = number of
    candidates removed by step (2), `gen_diams` / `gen_types` = diameter and ball type
    (the `s` of `ball`) per column.
    """
    kept, n_pruned = [], 0          # (x, diam, s) with x a ModuleMap R^1 -> R^n
    for diam in range(max_diam + 1):
        new = []
        for s in ball_types(diam):
            W, Wp = ball(diam, s), ball(diam + 2 * margin_rad, s)
            K = z2lin.kernel(f.expand(W)[0])
            K_Wp = np.zeros((f.n, len(Wp), K.shape[1]), dtype=BIT)
            K_Wp[:, np.searchsorted(Wp, W)] = K.reshape(f.n, len(W), K.shape[1])
            K_Wp = K_Wp.reshape(f.n * len(Wp), K.shape[1])
            prev = _hstack([translates(c[0], Wp) for c in kept], f.n * len(Wp))
            for j in z2_helpers.remove_image(prev, K_Wp)[1]:
                new.append((ModuleMap(W, K[:, j: j + 1], f.n).trimmed(), diam, s))
        keep = [True] * len(new)
        if same_diam_pruning:
            for i in reversed(range(len(new))):
                Wp = ball(diam + 2 * margin_rad, new[i][2])
                others = [c[0] for c in kept] + [c[0] for k, c in enumerate(new) if keep[k] and k != i]
                A = _hstack([translates(x, Wp) for x in others], f.n * len(Wp))
                keep[i] = bool(z2_helpers.remove_image(A, new[i][0].dense_columns(Wp))[1])
        n_pruned += keep.count(False)
        kept += [c for c, k in zip(new, keep) if k]

    counts = np.bincount([c[1] for c in kept], minlength=max_diam + 1).tolist()
    if verbose:
        print(f"level counts {counts}, pruned {n_pruned}")
    S = np.unique(np.concatenate([c[0].support for c in kept] or [[0]]))
    g = ModuleMap(S, _hstack([c[0].dense_columns(S) for c in kept], f.n * len(S)), f.n).trimmed()
    g.level_counts, g.n_pruned = counts, n_pruned
    g.gen_diams, g.gen_types = [c[1] for c in kept], [c[2] for c in kept]
    return g


def ball_span(g, diam, s=0):
    """
    F_2 matrix (in the flat (m, ball(diam, s)) layout) whose columns are all translates
    of the columns of g that fit inside ball(diam, s), i.e. a basis-free description of
    im(g) restricted to that ball.
    """
    W = ball(diam, s)
    return _hstack([translates(g.column(j), W) for j in range(g.n)], g.m * len(W))


def module_eq_on_ball(g, h, diam, s=0):
    """Do im(g) and im(h) agree on ball(diam, s)?  (as F_2 spans of translates)"""
    A, B = ball_span(g, diam, s), ball_span(h, diam, s)
    return not z2_helpers.remove_image(A, B)[1] and not z2_helpers.remove_image(B, A)[1]


# ------------------------------------------------------------------- infinite codes

def random_gens(n, m, diam, w, rng=None):
    """m random elements of R^n of weight w supported on ball(diam), as a map R^m -> R^n."""
    rng = np.random.default_rng() if rng is None else rng
    S = ball(diam)
    mat = np.zeros((n * len(S), m), dtype=BIT)
    for j in range(m):
        mat[rng.choice(n * len(S), size=w, replace=False), j] = 1
    return ModuleMap(S, mat, n).trimmed()


def generate_infinite_code(n, m, max_diam, init_diam, w_init, rng=None, f=None, margin_rad=0,
                           same_diam_pruning=True, verbose=False):
    """
    Random translation-invariant CSS code on the Cayley graph of F_2 with n qubits
    per vertex (see cayley_codes.md).

    m < n random Z-type generators of weight w_init supported on ball(init_diam)
    are assembled into f: R^m -> R^n (columns = the generators).  Then

        H_X = ker(f^dagger),   H_Z = ker(H_X^dagger),

    with ker = `syzygies(., max_diam, margin_rad, same_diam_pruning)`.  H_X collects the X operators commuting with all
    translates of the initial Z operators (x commutes with all translates of y iff
    y^dagger x = 0), and H_Z then collects *all* Z operators commuting with those,
    which generally contains more than the initial f.  Both are returned as maps into
    R^n (columns = stabilizer generators).  f can be passed in explicitly instead of
    being sampled.
    """
    if f is None:
        f = random_gens(n, m, init_diam, w_init, rng)
    H_X = syzygies(hermitian_transpose(f), max_diam, margin_rad, same_diam_pruning)
    H_Z = syzygies(hermitian_transpose(H_X), max_diam, margin_rad, same_diam_pruning)
    if verbose:
        print(f"f:   {f.n} Z gen(s), weights {f.weights()}, diams {f.diams()}")
        print(f"H_X: {H_X.n} gen(s) per vertex, level counts {H_X.level_counts}, "
              f"weights {H_X.weights()}, diams {H_X.diams()}")
        print(f"H_Z: {H_Z.n} gen(s) per vertex, level counts {H_Z.level_counts}, "
              f"weights {H_Z.weights()}, diams {H_Z.diams()}")
    return H_X, H_Z


# ------------------------------------------------------------------ compactification

class Compactification:
    """
    A finite quotient F_2 -> G, i.e. the 4-valent Cayley graph of a finite 2-generator
    group G, stored as the two permutations of [0, |G|-1] given by *right* multiplication
    with the generators a and b.  Element 0 is the identity.

    Only the two permutations are needed: since the right regular action is transitive and
    free, every group element is `reduce_word` of a word, and left multiplication (which is
    what acts on the stabilizers) is reconstructed from a BFS spanning tree.
    """

    def __init__(self, perm_a, perm_b):
        self.N = len(perm_a)
        self.R = {1: np.asarray(perm_a, dtype=int), 2: np.asarray(perm_b, dtype=int)}
        self.R[-1] = np.argsort(self.R[1])
        self.R[-2] = np.argsort(self.R[2])
        self._L = None

    def __repr__(self):
        return f"Compactification(|G|={self.N})"

    @classmethod
    def from_group(cls, e, a, b, mul):
        """Enumerate the group generated by a, b by BFS from the identity e (hashable
        elements, mul(x, y) = x*y) and build the two right-multiplication permutations."""
        elems, index, cols = [e], {e: 0}, {1: [], 2: []}
        i = 0
        while i < len(elems):
            for x, g in ((1, a), (2, b)):
                h = mul(elems[i], g)
                if h not in index:
                    index[h] = len(elems)
                    elems.append(h)
                cols[x].append(index[h])
            i += 1
        out = cls(cols[1], cols[2])
        out.elements = elems
        return out

    # -- words

    def reduce_word(self, w):
        """The G-element represented by the word w, as an index (0 = identity)."""
        g = 0
        for x in w:
            g = self.R[x][g]
        return int(g)

    def _bfs_tree(self):
        """BFS from the identity: returns (levels, letter) with levels[d] = the elements at
        distance d and letter[g] = the last letter of the BFS word reaching g."""
        dist = np.full(self.N, -1)
        letter = np.zeros(self.N, dtype=int)
        dist[0] = 0
        levels = [np.array([0])]
        while len(levels[-1]):
            new = []
            for x in GENS:
                img = np.unique(self.R[x][levels[-1]])
                img = img[dist[img] < 0]
                dist[img] = len(levels)
                letter[img] = x
                new.append(img)
            levels.append(np.concatenate(new))
        return levels[:-1], letter

    @property
    def L(self):
        """Left multiplication permutations: L[x][g] = index of pi(x) * g."""
        if self._L is None:
            levels, letter = self._bfs_tree()
            L = {x: np.zeros(self.N, dtype=int) for x in GENS}
            for x in GENS:
                L[x][0] = self.R[x][0]
            # g = h*y with h one step closer to the identity, so x*g = (x*h)*y
            for lvl in levels[1:]:
                for y in GENS:
                    sel = lvl[letter[lvl] == y]
                    par = self.R[-y][sel]
                    for x in GENS:
                        L[x][sel] = self.R[y][L[x][par]]
            self._L = L
        return self._L

    def left_perm(self, w):
        """The permutation g -> pi(w) * g."""
        p = np.arange(self.N)
        for x in reversed(w):
            p = self.L[x][p]
        return p

    # -- girth

    def find_girth(self):
        """
        Relative girth: the length of the shortest nonempty *reduced* word in a, b mapping
        to the identity of G (= the shortest cycle of the Cayley graph, counting a^2 = e
        style relations as cycles of length 2).

        BFS on the states (g, last letter), whose paths from the root are exactly the
        reduced words; the girth is the first time a state (identity, *) is reached.
        """
        li = {x: k for k, x in enumerate(GENS)}
        seen = np.zeros((self.N, 4), dtype=bool)
        frontier = {}
        for x in GENS:
            g = self.R[x][0]
            frontier[x] = np.array([g])
            seen[g, li[x]] = True
        d = 1
        while any(len(v) for v in frontier.values()):
            if any((v == 0).any() for v in frontier.values()):
                return d
            nxt = {}
            for y in GENS:
                img = np.unique(np.concatenate([self.R[y][frontier[x]] for x in GENS if x != -y]))
                img = img[~seen[img, li[y]]]
                seen[img, li[y]] = True
                nxt[y] = img
            frontier, d = nxt, d + 1
        return np.inf  # no relation at all (impossible for finite G)


def psl2p(p):
    """PSL(2, F_p) with the generators a = [[1,2],[0,1]], b = [[1,0],[2,1]] of the md."""
    def norm(M):
        M = tuple(x % p for x in M)
        return min(M, tuple((-x) % p for x in M))

    def mul(X, Y):
        return norm((X[0] * Y[0] + X[1] * Y[2], X[0] * Y[1] + X[1] * Y[3],
                     X[2] * Y[0] + X[3] * Y[2], X[2] * Y[1] + X[3] * Y[3]))

    return Compactification.from_group(norm((1, 0, 0, 1)), norm((1, 2, 0, 1)),
                                       norm((1, 0, 2, 1)), mul)
