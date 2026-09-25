# Implementation notes

State: **steps 1-3 done** (`core.py`, tested by `test_core.py`).

## Conventions fixed in step 1

- `R = F_2[F_2]`. Group elements = reduced words as int tuples: `1=a, -1=a^-1, 2=b, -2=b^-1`,
  identity `E = ()`. Helpers `w_mul`, `w_inv`, `w_str`, `star(l)` (depth-l star = reduced words
  of length <= l, BFS order; sizes 1, 5, 17, 53, 161, 485 for l=0..5 — note `cayley_codes.md`
  quotes 341/1365 for l=4/5, which counts non-reduced words).
- `ModuleMap` = a map `f: R^n -> R^m`, i.e. an `m x n` matrix over `R`, acting by left
  multiplication on column vectors. So `R^n` is a **right** R-module, composition is the matrix
  product `f @ g`, and **translation is right multiplication** by a group element (left
  multiplication does not commute with `f`).
- Storage: one overall `support` (list of `s` words, plus `index` dict) and a binary
  `mat` of shape `(s*m, n)` with row `i*s + k` = (output coordinate `i`, support element `k`).
  `f.blocks` is the same thing reshaped to `(m, s, n)`.
- Flat layout for module elements restricted to a finite support:
  `flat_index = coordinate * len(support) + support_index`. Used by `expand`, `dense_columns`
  and hence by all kernels.

## API

- `ModuleMap.from_entries(m, n, {(i,j): [words]})`, `.entry(i,j)`, `.column(j)`, `.trimmed()`
  (drop zero support elements), `.shifted(w)` (right translation; just relabels the support),
  `.is_zero()`, `repr` prints the matrix in `a/A/b/B` word notation.
- `.expand(support_in) -> (M, support_out)`: the F_2 matrix of `f` restricted to inputs supported
  on `support_in`; `support_out = supp(f) * support_in`.
- `.dense_columns(support)`: embed the columns of `f` into the flat layout of a bigger support.
- `compose(f, g)`.
- `syzygies(f, l_max)`: returns `g` with `f*g = 0`, plus `g.level_counts` = number of new
  generators found at each depth.

### Step 2

- `hermitian_transpose(f)`: `(f^dagger)_ji = (f_ij)^*`, `*` = antipode `g -> g^-1`. Implemented as
  `blocks.transpose(2,1,0)` plus inverting every support word. Anti-involution:
  `(f^dagger)^dagger = f`, `(fg)^dagger = g^dagger f^dagger`.
- `random_gens(n, m, l, w, rng)`: m random columns of R^n of weight exactly w on the depth-l star.
- `generate_infinite_code(n, m, l_max, l_init, w_init, rng=None, f=None, verbose=False)`
  -> `(H_X, H_Z)` with `H_X = syzygies(f^dagger)`, `H_Z = syzygies(H_X^dagger)`.
  Both are maps *into* `R^n` (columns = stabilizer generators, `R^n` = qubits per vertex).
- `star_span(g, l)`: F_2 matrix of all translates of g's columns fitting in the depth-l star;
  `module_eq_on_star(g, h, l)` compares two modules there.

### Step 3

- `Compactification(perm_a, perm_b)`: a finite quotient `pi: F_2 -> G`, stored as the two
  **right** multiplication permutations `R[x]` of `[0, |G|-1]`, index `0` = identity.
  `Compactification.from_group(e, a, b, mul)` builds them by BFS over any hashable group
  (`.elements` keeps the list); `psl2p(p)` = the `PSL(2, F_p)` family of the md.
- `reduce_word(w)`: apply the right permutations from the identity, giving the index of `pi(w)`.
- `comp.L[x]` (lazy property): the *left* multiplication permutations, `L[x][g] = pi(x)*g`.
  These are what act on the stabilizers (the map `f` multiplies the support from the left, see
  `expand`), and they are not directly given by the input data. Reconstructed from a BFS spanning
  tree: if `g = h*y` with `h` one step closer to the identity, then `x*g = (x*h)*y`, i.e.
  `L[x][g] = R[y][L[x][h]]` — one vectorized pass per BFS level. `left_perm(w)` composes them.
- `find_girth()`: BFS on the states `(g, last letter)` (4|G| of them), whose paths from the root
  are exactly the reduced words; the girth is the first distance at which a state
  `(identity, *)` appears. So this is the *algebraic* girth = shortest nonempty reduced word in
  `ker(pi)`, which counts e.g. `a^2 = e` as a cycle of length 2. ~1.6 s for |G| = 515100.
- `ModuleMap.compactify(comp)`: the ordinary `(m|G|) x (n|G|)` binary matrix, flat layout
  `index = coordinate * |G| + group index`. For each support word `u`, row block
  `left_perm(u)` gets the coefficients; `^=` (not `=`) because distinct words of `F_2` can
  collapse onto the same element of `G`. Fast (0.06 s for |G| = 1092, n = 5).
- Compactification is functorial and turns the dagger into the plain transpose:
  `compactify(f g) = compactify(f) compactify(g)` and `compactify(f^dagger) = compactify(f)^T`
  (because `pi(u)^-1 d = g <=> d = pi(u) g`). Hence `H_X^dagger H_Z = 0` immediately gives the
  finite CSS commutation `compactify(H_X)^T compactify(H_Z) = 0` — checked in `test_compactify`.
- Girths found for `psl2p(p)`: p=5,7,11,13,31,61,101 -> |G|=60,168,660,1092,14880,113460,515100
  with girth 5,6,9,9,12,15,14 (logarithmic, and not monotone in p).
- Observed so far: for the small random codes tried (n <= 5, l_max = 2, |G| = 60 or 1092) the
  compactified code has `k = 0`, i.e. the `n|G|` qubits are used up by the stabilizers with no
  dependencies gained from the quotient. Consistent with `k_X + k_Z = n` per vertex; to get
  logicals one needs `m < n` in the Euler-sum sense (few stabilizers per vertex), not just few
  initial generators.

### Why the dagger is the right notion of commutation (checked by `test_pairing`)

Translation is right multiplication, so the Z operator module generated by a column `y` is
`{y c : c in R}`. With `<x, y> = int(x^* y)` (`int` = coefficient of the identity) equal to the
overlap parity of `x` and `y`, one gets `<y v, x u> = ` coefficient of `y^dagger x` at `v u^-1`.
Hence **all** translates of `y` commute with all translates of `x` iff `y^dagger x = 0` as a ring
element. So `H_X = ker(H_Z^dagger)` is exactly "all X operators commuting with the Z stabilizers".
`H_X^dagger H_Z = 0` implies `H_Z^dagger H_X = 0` by anti-multiplicativity, so the third
annihilator automatically contains the first; `test_infinite_code` checks equality on the star.

### Empirical: stabilizers per vertex (`test_euler_sum`)

`k_X + k_Z = n` (the md's `m = n`) holds for most random samples, but not always: the raw counts
can overshoot because the returned generators need not be R-independent (see below). The Euler sum
`k_X - syz(H_X) + k_Z - syz(H_Z)` was `= n` in 72/72 random trials, including all overshooting
ones. So the overshoot is real redundancy, not a violation of the claim.

Note `F_2[F_2]` is a free ideal ring (Cohn), so every submodule of a free module is free and the
global dimension is 1: the *true* syzygy module has a basis and second syzygies vanish. Any second
syzygy we compute is therefore an artifact of our generating set, never of the ring. (Contrast
`F_2[Z^d]`, global dimension d; and note `F_2[PSL(2,Z)] = F_2[C_2 * C_3]` contains
`F_2[C_2] = F_2[x]/(x^2)`, which has *infinite* global dimension in characteristic 2 — for that
group the Euler-sum recipe of the md will not terminate, and the rate needs a different handle.)

Diagnosis of the observed redundancy: a generator accepted at depth `l` is never revisited, and at
depth `l` the combination witnessing its redundancy uses translates of *other* depth-`l`
generators, which stick out of the depth-`l` star. In every overshooting case inspected, the
relation had a **unit** coefficient (a single group element — `F_2[F_2]` has only trivial units),
i.e. one generator is literally an R-combination of translates of the others, and testing that
directly inside the depth-`l_max` star already detects it. So a cheap final pruning pass (for each
generator, is it in the R-span of the other generators' translates inside the star?) would fix all
cases seen so far. Deletion can fail in principle if *every* relation has all coefficients
non-units; then a basis still exists but its elements are R-combinations of the old generators, so
one would have to change basis rather than drop generators. Not implemented — deliberately left
as is, since for groups other than `F_2` genuinely longer resolutions are possible. With `l_init = 1` and
`w_init ~ n`, `H_X` is frequently empty for `m >= n/2`, which makes `H_Z = identity` (a trivial
code with no stabilizers) — for interesting codes keep `m` small relative to `n`.

## Deviation from the sketch in cayley_codes.md (syzygies)

The md says to quotient the depth-`l` kernel only by translates of the generators from depths
`m < l`. That does not give *independent* generators: e.g. `f = (1+a, 1+a^-1)` has kernel
generated by `x = (a^-1, 1)`, and `x*a = (1, a)` is also supported on the depth-1 star, so the
depth-1 kernel is 2-dimensional while the syzygy module is free of rank 1.
Implemented instead: process the depth-`l` kernel columns greedily, and after accepting a
generator `x`, quotient by *all* translates `x*w` whose support still fits in the depth-`l` star
(for every accepted generator, from this depth or lower). Candidate `w`'s are found as
`u^-1 * star(l)` for one `u` in `supp(x)`, then filtered — cheap. This makes the returned
generators independent in the sense that no one of them is in the R-span of the others
restricted to the window where the computation can see it.

`g` is **not** guaranteed to be injective as a map of R-modules: a relation between the returned
generators may only become visible on a star larger than `l_max`. This does happen in practice —
`test_euler_sum` finds random codes where `syzygies(H_X, l_max)` is nonzero. Removing that
redundancy = the free-resolution/Euler-sum computation of the "Encoding rate" section of the md.
`test_core.py` checks `f*g = 0` and that the translates of `g`'s columns exactly span the kernel
of `f` on every star up to `l_max`.

## Next

- Free resolution / Euler sum to get the true number of stabilizers per vertex (and to prune
  redundant generators).
- Distance of the compactified codes; search for compactifications (girth) and for infinite codes
  with `k > 0` after compactification.
