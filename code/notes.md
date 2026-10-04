# Implementation notes

State: **steps 1-3 done** (`core.py`, tested by `test_core.py`). Step 4 = design only: interface for
general groups proposed in `group_implementation.md` (open questions there). Step 5 done: integer-indexed
ball (the Q3 "performance option", still F_2 only) and pruning of syzygy generators. Since then:
syzygies in half-integer radius steps, all window sizes given as ball *diameters* (see
"Syzygies in diameter steps" below).

## Conventions fixed in step 1

- `R = F_2[F_2]`. Words = reduced int tuples `1=a, -1=a^-1, 2=b, -2=b^-1`, identity `E = ()`,
  helpers `w_mul`, `w_inv`, `w_str` (only used to build the ball and for I/O).
- **Since step 5, group elements are ints**: the index in the global BFS enumeration `BALL` (class
  `Ball`, grown lazily, indices stable under growth, 0 = identity). The radius-l ball around e is the
  range `_star(l) = arange(BALL.size(l))` (internal; `BALL.size/grow` also take radii); sizes 1, 5,
  17, 53, 161, 485 for l=0..5 (`cayley_codes.md` quotes 341/1365 for l=4/5, which counts
  non-reduced words). Arrays: `length`, `parent`/`letter`
  (BFS tree, `g = parent[g] * gens[letter[g]]`), `R[k, g] = g * gens[k]` (-1 outside the ball, with a
  sentinel column `R[:, -1] = -1` so -1 propagates through fancy indexing), `inv[g]`.
  `BALL.mul(A, B)` = full product table `A[i]*B[j]`, computed along the BFS tree of `B`, one fancy
  indexing step per level; -1 if outside the ball (callers `grow` first when they need exact
  products; `translates` relies on -1 to mean "doesn't fit"). `BALL.elem(word)`, `BALL.word(g)`
  convert. Only `mul`/`inv`/gens on normal forms are used to build the ball, so it generalizes
  directly to other groups (`group_implementation.md`).
- **Window sizes are diameters.** `ball(diam, s=0)`: even diam = radius-diam/2 ball around e;
  odd diam = radius-diam/2 ball around the midpoint of the edge `{e, x}`, `x = EDGE_GENS[s]`
  (`a` or `b`), i.e. `_star(diam//2) u _star(diam//2)*x` (distance `d(g,h) = |g h^-1|`, invariant
  under right multiplication). `ball_types(diam)` = the valid `s` (2 for odd diam). Sizes 1, 2, 5,
  8, 17, 26, 53 for diam=0..6 (`3^(l+1)-1` for odd diam = 2l+1). All public `l`/`l_max`/`l_init`
  parameters became `diam`/`max_diam`/`init_diam` (`l = diam/2`, so old `l_max = 3` is `max_diam = 6`).
- F_2 matrices use dtype `BIT = np.uint8` (8x less memory; parity survives uint8 wraparound).
- `ModuleMap` = a map `f: R^n -> R^m`, i.e. an `m x n` matrix over `R`, acting by left
  multiplication on column vectors. So `R^n` is a **right** R-module, composition is the matrix
  product `f @ g`, and **translation is right multiplication** by a group element (left
  multiplication does not commute with `f`).
- Storage: one overall `support` (**sorted** int array of `s` elements, sorted by the constructor) and a binary
  `mat` of shape `(s*m, n)` with row `i*s + k` = (output coordinate `i`, support element `k`).
  `f.blocks` is the same thing reshaped to `(m, s, n)`.
- Flat layout for module elements restricted to a finite support:
  `flat_index = coordinate * len(support) + support_index`. Used by `expand`, `dense_columns`
  and hence by all kernels.

## API

- `ModuleMap.from_entries(m, n, {(i,j): [words or ints]})`, `.entry(i,j)` (returns words),
  `.column(j)`, `.trimmed()` (drop zero support elements), `.shifted(w)` (right translation, w int
  or word; relabels and re-sorts the support),
  `.is_zero()`, `repr` prints the matrix in `a/A/b/B` word notation.
- `.expand(support_in) -> (M, support_out)`: the F_2 matrix of `f` restricted to inputs supported
  on `support_in` (int array); `support_out = supp(f) * support_in`. Fully vectorized.
- `.dense_columns(support)`: embed the columns of `f` into the flat layout of a bigger (sorted) support.
- `.diams()`: support diameter of each column.
- `translates(x, window)`: all translates of a single column fitting in a window (sorted int
  array, e.g. `ball(diam, s)`), as an F_2 matrix. `ball_span(g, diam, s=0)` = hstack over the columns.
- `compose(f, g)`.
- `syzygies(f, max_diam, margin_rad=0, same_diam_pruning=True)`: returns `g` with `f*g = 0`, plus `g.level_counts[d]` =
  number of kept generators of diameter d, `g.n_pruned`, and per column `g.gen_diams`,
  `g.gen_types` (the `s` of its ball). See "Syzygies in diameter steps" below.

### Step 2

- `hermitian_transpose(f)`: `(f^dagger)_ji = (f_ij)^*`, `*` = antipode `g -> g^-1`. Implemented as
  `blocks.transpose(2,1,0)` plus inverting every support word. Anti-involution:
  `(f^dagger)^dagger = f`, `(fg)^dagger = g^dagger f^dagger`.
- `random_gens(n, m, diam, w, rng)`: m random columns of R^n of weight exactly w on `ball(diam)`.
- `generate_infinite_code(n, m, max_diam, init_diam, w_init, rng=None, f=None, margin_rad=0, same_diam_pruning=True, verbose=False)`
  -> `(H_X, H_Z)` with `H_X = syzygies(f^dagger)`, `H_Z = syzygies(H_X^dagger)`.
  Both are maps *into* `R^n` (columns = stabilizer generators, `R^n` = qubits per vertex).
- `ball_span(g, diam, s=0)`: F_2 matrix of all translates of g's columns fitting in `ball(diam, s)`;
  `module_eq_on_ball(g, h, diam, s=0)` compares two modules there.

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

## Syzygies (step 5): candidates + pruning  (superseded by the next section)

Written in the old radius terms (`l` = star depth).

1. Candidates, depth by depth: an F_2 basis of the kernel on the depth-l star (bitgauss nullspace),
   minus the vectors in the span of the translates of the *lower-depth* candidates (one
   `remove_image` per depth). Going up in depth keeps the translates of small generators from
   mixing into the deeper candidates. Same-depth translates (e.g. `x` and `x*a`) both stay candidates.
2. Pruning: going from the deepest candidate to the shallowest, drop `x` if it lies in the span of
   the translates (inside the depth-`l_prune = l_max + l_margin` star, `l_margin=0` by default) of all *other* remaining
   candidates. This removed every raw-count overshoot in `test_euler_sum` (0 instead of 1) and 5
   redundant generators in 30 random cross-checks against the step-3 code (same R-modules otherwise).

The step-3 code instead accepted kernel vectors one at a time within a depth, immediately
quotienting by their translates. Pruning makes that redundant (anything dependent on the earlier
generators is also dependent on "all others"), and an experiment confirmed identical generator
counts and depths in 20 random cases for: greedy+prune, the per-depth filter+prune above, and even
the full kernel of every depth + prune with no filter. The only difference is the number of
candidates and hence of pruning rref's: at l_max=4, (6,2,6) needs 4 / 8 / 304 candidates and
0.01 / 0.02 / 12 s. So we use the per-depth filter.

Why same-depth translates matter: on the tree, a generator of support diameter D first appears at
depth ceil(D/2); for odd D its center is an edge midpoint, so *two* translates fit in that star (e.g.
`(a^-1, 1)` and `(1, a)` for `f = (1+a, 1+a^-1)`). In 200 random cases (l_max=3): 627 candidates after
the per-depth filter, 122 of them same-depth translate duplicates, 47 more removed by pruning,
458 final (133 with odd D). (Re-checked later on 98 problems: of 43 pruned candidates, 41 were
redundant via same-depth candidates alone and 2 via same-depth plus shallower ones; *none* needed
a deeper candidate, so "pruning across depths" never removed a shallow generator.) Pruning as one rref (columns = all translates, deepest candidate
first, each candidate tested only against those after it) gives 476 instead of 458 (differs in 14/200
cases): it misses generators that need both shallower and deeper ones. So the pruning stays sequential
(one rref per candidate; cheap since #candidates ~ 1.4 x #generators).

**Window subtlety.** Pruning keeps the generated R-*module* unchanged (`x = sum y_k r_k` exactly),
but not the span of translates *fitting inside* a given star: `x*w` may fit while the witnesses
`y_k r_k w` stick out. Verified: the deficit is 81/243/729 boundary vectors on stars of depth
6/7/8, i.e. it lives on the boundary sphere. So the guarantee is "kernel on the depth-`l_max` star
lies in the R-module of `g`"; tests check this on the depth-`2*l_max` star (`check_syzygy`,
`test_infinite_code`). Compare modules by testing *generators* of one against translates of the
other, not by comparing windowed spans.

Pruning window (200 syzygy problems from 100 random codes, l_max=3): `l_margin` = 0/1/2/3 gives
461/460/460/460 generators, nonzero 2nd syzygies (on the depth-3 star) in 1/0/0/0 cases, 0.3/0.7/4.3/37 s.
Deletion-minimal != minimum: `{1+a, 1+a+a^2}` generates R (`1 = (1+a+a^2) + (1+a)a`) but neither is
redundant at any window. Guess for other groups: hyperbolic ~ margin of half the longest relator;
Z^d may need larger margins (Groebner-type degree bounds) but balls are cheap there.

## Syzygies in diameter steps

Idea: also use the odd-diameter balls (radius l + 1/2 around an edge midpoint, one shape per
generator; each contains two radius-l balls, and each radius-(l+1) ball contains two of them per
generator). Process `diam = 0, 1, ..., max_diam`; at odd diam both edge balls.

Why this removes the same-depth duplicates (tree argument): a finite set of diameter D has a
unique centre (vertex for even D, edge for odd D) and lies in the diameter-D ball around it. If
`S` lies in a diameter-d ball `B` and has diameter `D < d`, its own diameter-D ball lies inside `B`
(the centres are at distance <= (d-D)/2). So by induction anything in the kernel on `B` of smaller
diameter is spanned by translates of earlier candidates, and every *new* candidate has diameter
exactly d with centre = centre of B. A translate `x*w`, `w != 1`, moves the centre (free action, and
no element of F_2 flips an edge), so the only translate of a new candidate fitting in its ball is
itself, and odd-diameter candidates of type a never fit in a b-edge ball. Hence the per-ball rref
already makes the same-diameter candidates independent. (Breaks for generators of order 2 — the
edge is flipped, two translates fit — and for non-tree Cayley graphs, e.g. Z^2, where centres are
not unique.)

Algorithm (`syzygies(f, max_diam, margin_rad=0, same_diam_pruning=True)`): for each ball `W` of
diameter diam let `W+ = ball(diam + 2*margin_rad)` with the same centre, i.e. the radius enlarged
by the integer `margin_rad` (a margin in diameter would have to be even for `W+` to be concentric).
1. Candidates: F_2 kernel basis on `W`, minus the span of the translates *inside W+* of the kept
   smaller-diameter generators (one `remove_image` per ball). This is the lower-diameter pruning;
   no separate candidate list is kept. A dropped vector never reappears at a larger diameter: a
   translate of it fitting in a larger ball lies in a concentric sub-ball (tree lemma above), whose
   enlargement by the margin lies inside the larger ball's `W+`.
2. If `same_diam_pruning`: drop a new candidate x (last found first) if it is in the span of the
   translates inside its `W+` of the smaller generators and the remaining other same-diameter
   candidates. Not switched off for `margin_rad = 0`: for F_2 it is a no-op then, but for other
   groups a ball can contain several translates of a candidate.

Smaller-diameter generators are never removed (user's choice: like a Groebner basis, keep generators
with distinct minimal "leading" balls rather than a minimal generating subset; replacing a small
generator by big ones could make the higher syzygies less local). `g.n_pruned` counts only the
step-2 removals.

Checked in `check_syzygy`: `g.diams() == g.gen_diams` (every generator has diameter exactly its
level) and each generator lies in `ball(gen_diam, gen_type)`; kernel on the diameter-max_diam
ball(s) is covered by `ball_span(g, 2*max_diam)`; no generator is in the span of the smaller ones
(plus same-diameter ones if `same_diam_pruning`) inside its `W+`, nor of all others of at most its
diameter inside its own ball.

Experiments (same random codes as before, n in 3..6, init_diam = 2):
- Old radius-step code vs diameter steps at max_diam = 6 (= l_max 3), ~98 problems per seed, 4
  seeds: identical R-modules (checked on the depth-2*l_max star); old needed 263/270/258/267
  candidates and kept 220/226/210/220, the diameter steps produce 220/225/210/220 candidates and
  pruning removes none. The one difference (226 vs 225) is the known case where the old code
  needed `l_margin = 1`.
- New code, ~200 problems each at max_diam = 5 and 6: `margin_rad` = 0, 1, 2, with and without
  `same_diam_pruning`, all give 445 / 450 generators (same as the margin-free filter), 0
  same-diameter prunes, 0 problems with nonzero second syzygies (computed with the same
  max_diam). Time for all ~200: 0.4-0.6 s at margin_rad 0, 0.5-0.9 s at 1, 1.4-3.3 s at 2. So for F_2 the generators come out R-independent in every case seen;
  not proven (a relation among same-diameter generators with different centres, cancelling each
  other's extreme points, is not ruled out). `margin_rad` defaults to 0.

## Performance (step 5)

The integer ball did not make syzygies faster at the sizes we use (n <= 8, l_max <= 5: 0.05–0.2 s
either way): the time is spent in dense rref (bitgauss) and in allocating the dense expansion
(f.expand on star(6) with n=6 is ~10^4 x 10^4). uint8 halved expand time at l=6; compactify is
4x faster (vectorized over entries). The point of the refactor is mainly the generic structure.
Next performance step if needed: avoid materializing the dense expansion (sparse / bit-packed).

## Next

- Free resolution / Euler sum to get the true number of stabilizers per vertex (and to prune
  redundant generators).
- Distance of the compactified codes; search for compactifications (girth) and for infinite codes
  with `k > 0` after compactification.
