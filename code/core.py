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

    def compactify(self, comp):
        """
        Push f through a quotient F_2 -> G (a `Compactification`), giving the ordinary
        binary matrix of shape (m*|G|, n*|G|) in the flat layout
        index = coordinate * |G| + group element index.
        """
        N = comp.N
        out = np.zeros((self.m * N, self.n * N), dtype=int)
        V = out.reshape(self.m, N, self.n, N)
        cols = np.arange(N)
        for k, u in enumerate(self.support):
            rows = comp.left_perm(u)  # rows[g] = index of pi(u) * g
            for i, j in zip(*np.nonzero(self.blocks[:, k, :])):
                V[i, rows, j, cols] ^= 1  # ^=: several support words can collapse in G
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
