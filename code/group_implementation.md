# Generalizing from F_2 to other infinite groups — interface design (step 4)

Nothing is implemented yet. This file lists (1) what the current code actually uses from `F_2`,
(2) a proposed interface for a general group `G_inf`, and (3) open questions. Questions are marked
**Q1, Q2, …**. Each has a recommendation; please confirm or overrule it.

## 1. What the current code uses from F_2

| place in `core.py` | F_2-specific ingredient | general replacement |
|---|---|---|
| element representation | reduced word = tuple of ints; hashable and **canonical** | a hashable normal form `g` |
| `w_mul`, `w_inv` | free reduction | `mul(g, h)`, `inv(g)` |
| `E`, `GENS = (1,-1,2,-2)` | 4 letters, `-x` = inverse letter | `identity`, symmetric generating set `letters` with `inv_letter` |
| `star(l)` | reduced words of length `<= l` | ball `B_l` = BFS in the Cayley graph from `identity` |
| `len(w)` (`depths`, `_sort_words`) | length of the reduced word | word length `length(g)` = BFS distance |
| `_sort_words` | order by `(len, word)` | BFS order of the ball (any fixed total order works) |
| `from_entries`, `repr` | the words themselves | `evaluate(word)`, `fmt(g)` |
| `hermitian_transpose` | `w_inv` | `inv` |
| `expand`, `compose`, `_valid_shifts`, `shifted` | `w_mul` | `mul` |
| `syzygies`, `star_span`, `random_gens` | `star` | `ball` |
| `Compactification` | any two perms define a quotient of F_2 | perms must satisfy the **relators** of `G_inf` |
| `reduce_word`, `left_perm(u)` | `u` *is* a word | need a word for `g`: `word(g)` |
| `find_girth` | BFS over states `(g, last letter)` = reduced words | needs a finite-state description of normal forms (see Section 4) |
| Euler sum (planned) | `F_2[F_2]` is a FIR, gl.dim 1 | depends on the group; infinite for torsion (Section 5) |

So the algebra (`compose`, `expand`, `syzygies`, `hermitian_transpose`, `compactify`) only needs
`mul`, `inv`, `identity`, the generating set and a hashable normal form. Everything else
(balls, lengths, ordering) can be derived generically.

## 2. Word problem vs normal form (Q1)

Your suggestion: the interface provides `is_trivial(word)`. This is enough in principle, but in
practice it is not what the code needs: building balls, supports and the `index` dicts requires
**hashing** elements. With only `is_trivial`, deduplicating `N` elements means `O(N^2)` calls of
`is_trivial(u^-1 v)`, and every `w_mul(u, v)` lookup in a support becomes a linear search.

Proposed instead: the group provides a **canonical hashable normal form** for elements, plus `mul`
and `inv` on normal forms. This is equivalent in power to a solvable word problem for all groups we
care about, and your examples fit directly:
- `SL(2,Z)`: the integer matrix itself, as a tuple `(a, b, c, d)`.
- `PSL(2,Z)`: the matrix normalized by sign (as `psl2p` already does mod p), or the alternating
  normal form in `C_2 * C_3`.
- `F_2`: the reduced word (current code).
- `Z^d`: the integer vector.
- surface/hyperbolic groups: shortlex-irreducible words of a confluent rewriting system
  (Knuth–Bendix), see Section 6.

`is_trivial(word)` then comes for free: `evaluate(word) == identity`.

**Q1:** OK to require a normal form instead of only `is_trivial`? (Recommendation: yes.)

## 3. Proposed interface

```python
class InfiniteGroup(ABC):
    # --- must be provided ------------------------------------------------
    letters: tuple            # symmetric generating set, e.g. (1, -1, 2, -2)
    identity                  # normal form of e (hashable)
    def inv_letter(self, x)   # inverse letter (x itself for an involution)
    def gen(self, x)          # normal form of the generator x
    def mul(self, g, h)       # normal form of g*h
    def inv(self, g)          # normal form of g^-1
    relators: list            # words equal to e; defines which finite actions are allowed (Q5)

    # --- optional, with generic defaults ---------------------------------
    def fmt(self, g) -> str           # pretty printing (default: geodesic word(g) in aAbB… notation)
    def word(self, g)                 # some word for g (default: BFS-tree geodesic from the ball cache)
    def normal_form_automaton(self)   # finite-state acceptor of unique geodesic words, for find_girth (Section 4)
    torsion: bool                     # information for the Euler sum (Section 5)

    # --- derived, implemented once in the base class --------------------
    def evaluate(self, word)          # fold mul over gen(x)
    def ball(self, l)                 # list of elements at distance <= l, BFS order (cached, grows lazily)
    def length(self, g)               # BFS distance (from the cache; extends the cache if needed)
    def sort_key(self, g)             # BFS index -> replaces _sort_words
    def is_trivial(self, word)
```

`ModuleMap` gets a `group` attribute (or the group is a module-level/global setting, **Q8**), and
`w_mul`, `w_inv`, `star`, `len(w)`, `_sort_words` in `core.py` are replaced by the group methods.
`FreeGroup(k)` reproduces the current code exactly (and can keep the fast tuple implementation),
which gives a regression test: all existing tests must pass with `G_inf = FreeGroup(2)`.

### Performance option: integer-indexed ball (Q3) — **implemented for F_2 (step 5)**, see `Ball` in `core.py`

All computations happen inside a ball `B_L` of fixed radius. Instead of hashing normal forms in
every `mul`, one can enumerate `B_L` once (generic code, using only the interface above), index it
by integers `0..|B_L|-1` in BFS order, and store
- `R[x]` = right multiplication by each letter `x` as an int array on `B_{L-1} -> B_L`,
- `inv` as an int array,
- products `u*v` computed from `R` by walking along `word(v)` — vectorized over all `u` at once,
  exactly like `Compactification.left_perm`.

Supports then become int arrays, and `expand`/`compose`/`_valid_shifts` become numpy fancy
indexing instead of Python loops over tuples. This is the "partial Cayley graph" analogue of the
current `Compactification` class, so a lot of code can be shared.

**Q3:** Do this refactor now (while generalizing), or first generalize with hashable normal forms
and optimize later? (Recommendation: generalize first with normal forms — balls of radius 4 or
5 are only a few hundred to a few thousand elements; refactor to int indices only if profiling
shows it matters. For amenable groups like `Z^2`, balls are small and we can go to large `l`.)

## 4. Compactification and girth

### 4.1 Quotients must satisfy the relators (Q5)

For `F_2` any two permutations define a quotient. For general `G_inf = <S | relators>`, a
`Compactification` is only valid if the permutations satisfy every relator. Plan:
`Compactification(group, perms)` checks `left_perm(r) == identity` for each relator `r`. Then
`compactify` works unchanged, with `left_perm(u)` computed along `group.word(u)` (any word
for `u` gives the same permutation, precisely because the relators hold).

### 4.2 Finite G_inf-sets instead of finite groups (Q4)

Observation: `compactify` only uses the *left action* of `G_inf` on a finite set `X`
(`x -> pi(u) x`), and the two properties that make the CSS code work —
`compactify(fg) = compactify(f) compactify(g)` and `compactify(f^dagger) = compactify(f)^T` —
hold for **any** finite set `X` with a `G_inf`-action by permutations (`u^-1 x = y <=> x = u y`).
Translation invariance on `X` is not needed for the finite code.

So instead of a finite quotient group `G` (Cayley graph) we can use any finite **Schreier graph**
`G_inf / H` for a finite-index subgroup `H` (not necessarily normal). This is exactly the `S_n`
sampling of `cayley_codes.md`, but without taking the generated subgroup: the code lives on the
`n` points directly, instead of on `|<rho(a), rho(b)>|` elements, which can be as large as `n!`.
It gives many more and much smaller compactifications. The girth notion becomes a per-point
injectivity radius (shortest nontrivial-in-`G_inf` word fixing some point).

The current class would then simply store the left-action permutations of the letters (for a
Cayley graph these are the `L[x]` already reconstructed in step 3).

**Q4:** Generalize `Compactification` to finite `G_inf`-sets (Schreier graphs)? (Recommendation:
yes — it contains the group case as the regular action and costs nothing.)

### 4.3 Relative girth for general G_inf (Q6)

Definition (as in the md): length of the shortest word `w` with `pi(w) = e` in `G` but `w != e`
in `G_inf`. The current `find_girth` BFS over `(g, last letter)` works because the reduced words
are exactly the words accepted by a finite automaton whose state is the last letter, and they are
in bijection with `F_2`. This generalizes verbatim to any group with a **finite-state acceptor of
unique geodesic normal forms** (`normal_form_automaton()` above): BFS over states
`(g in G, automaton state)`, girth = first length at which a nonempty accepted word reaches
`(e, accepting state)`. Available for:
- `F_2`, free products such as `PSL(2,Z) = C_2 * C_3` (state = last syllable type),
- `Z^d` (e.g. state = last coordinate direction and sign, for the sorted normal form),
- all hyperbolic (and automatic) groups, e.g. surface groups, via a shortlex automaton (kbmag).

Generic fallback without an automaton: grow `B_r(G_inf)` and its image in `G`, and find the first
collision `pi(u) = pi(v)`, `u != v`; then the kernel contains `u^-1 v` of length `<= 2r`, and
the girth is `min length(u^-1 v)` over collisions at the first such `r`. Cost ~ `|B_{girth/2}|`,
which is about `sqrt(|G|)` for the families we care about — fine.

Note on the convention: for `F_2` the current code counts `a^2 = e` in `G` as girth 2. For a group
with torsion where `a^2 = e` already in `G_inf`, that word is trivial and does not count.

**Q6:** Implement the generic collision fallback first and the automaton version only for groups
where it's easy (F_2, free products)? (Recommendation: yes.)

## 5. Torsion, Euler sum and syzygies (Q7)

The syzygy algorithm itself is group-independent. What changes:
- With torsion (`PSL(2,Z)`, `SL(2,Z)`, triangle groups), `F_2[G_inf]` contains
  `F_2[C_2] = F_2[x]/(x^2)` with infinite global dimension in characteristic 2, so free
  resolutions can be infinite and the Euler-sum count of stabilizers per vertex need not
  terminate. (Over `F_2`, `C_3` is harmless — `F_2[C_3]` is semisimple — only even torsion causes
  trouble. So `C_3 * C_3` or odd triangle groups like `(3,3,5)` are less bad than `C_2 * C_3`;
  odd-order torsion still gives finite projective resolutions over `F_2`.)
- For torsion-free groups of cohomological dimension `d` (`Z^d`, surface groups: `d = 2`)
  resolutions have length `<= d`, so second syzygies are real, not artifacts of our generating set
  (unlike for `F_2`).

**Q7:** For groups with even torsion, what should the Euler-sum code report — the truncated
alternating sum up to a fixed resolution length (with a warning), or should we restrict the
Euler-sum analysis to torsion-free / odd-torsion groups? Also: is the "pruning to independent
generators" (notes.md) still wanted for groups where non-free submodules exist?

## 6. Which groups to implement first (Q2, Q9)

| group | normal form | gens / valency | torsion | notes |
|---|---|---|---|---|
| `FreeGroup(k)` | reduced word | `2k` | no | regression test against current code |
| `Z^d` | int vector | `2d` | no | amenable: expect `a = n` exactly; polynomial balls, large `l` ok |
| `PSL(2,Z) = C_2 * C_3` | alternating syllables or ±matrix | **3** (`s`, `t`, `t^-1`) | 2 and 3 | hyperbolic; the Cayley graph is 3-valent |
| `SL(2,Z)` | integer matrix | depends on gens | 4 and 6 | not torsion-free; centre `±1` |
| `C_3 * C_3` | alternating syllables | 4 | 3 only | torsion harmless over `F_2` |
| `F_2 x F_2`, `F_2 x Z`, … | pair of normal forms | sum | inherited | generic direct-product combinator |
| genus-`g` surface group | shortlex word (Knuth–Bendix) | `4g` | no | cd 2, hyperbolic; needs rewriting system |

**Q2 (contradiction/choices in the md):**
- The md says `PSL(2,Z)` has relations `a^3`, `b^2`, and later in the compactification section
  `rho(a)^2 = 1`, `rho(b)^3 = 1`. Which convention?
- With an involution generator the Cayley graph is **not 4-valent**: `PSL(2,Z)` with `s^2 = t^3 = e`
  gives `{s, t, t^-1}`, 3-valent. The current code hardcodes 4 letters `(1,-1,2,-2)` in `GENS`,
  `Compactification`, `find_girth`. Proposal: allow an arbitrary symmetric generating set (with
  involutions as self-inverse letters), so valency = number of letters. Alternatively keep 4
  letters by choosing generators of infinite order (e.g. `PSL(2,Z)` with `T = [[1,1],[0,1]]`,
  `U = [[1,0],[1,1]]` — but these generate `PSL(2,Z)` only together with relations that are not a
  simple `a^i`, `b^j` presentation, and stars look different). Which do you want?

**Q9:** Which groups do you want first? (Recommendation: `FreeGroup(k)` refactor + `Z^2` as a
sanity check of the generic code, then `PSL(2,Z)` and `C_3 * C_3`, then a genus-2 surface group.)

Surface group normal forms: there is a known finite confluent rewriting system for surface groups
(Le Chenadec), so a small pure-Python string rewriting engine suffices; alternatively sympy's
`FpGroup`/`RewritingSystem` (Knuth–Bendix) or kbmag via GAP. Neither sympy nor GAP is currently in
`.venv`. Finding finite quotients/Schreier graphs of finitely presented groups: sympy's
`low_index_subgroups` (Todd–Coxeter) or GAP's `LowIndexSubgroupsFpGroup`; or random permutation
sampling with a relator check (the md's approach), which needs nothing extra.

**Q10:** OK to add `sympy` (and possibly GAP via subprocess) as dependencies, or prefer
self-contained implementations?

## 7. Code organisation (Q8)

- New file `code/groups.py`: the ABC `InfiniteGroup` with the generic derived methods (ball,
  length, evaluate, word, collision-girth) and the concrete groups.
- `core.py`: `ModuleMap(group, support, mat, m)`; `compose`, `syzygies`, … read the group from
  their arguments (assert both maps share it). `Compactification(group, perms)` checks relators.
- Free-function helpers `w_mul`, `star` etc. stay as thin aliases for `FreeGroup(2)` so the
  notebooks keep working, or get removed.

**Q8:** Group stored on each `ModuleMap` (recommended; allows mixing groups in one session), or a
global "current group"? Keep the old `w_mul`/`star` helpers for the notebooks?
