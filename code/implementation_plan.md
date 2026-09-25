# Implementation plan

These notes contain instructions for Claude to implement the ideas in cayley_codes.md step by step.

### General instructions
- Implement in python, make use of specialized libraries when possible (but not to replace short code).
- use the virtual environment .venv for testing
- this is a research project. no need for it to be super polished and test all exceptions or verify user inputs. no need to define getters and setters for everything etc. keep it concise.
- ask questions before implementing if anything is unclear
- avoid explicit python loops unless it doesn't matter for performance. try to use numpy where possible
- keep for yourself compact notes about the state of the implementation
- maintain a .gitignore covering all files that aren't "sources" in the usual sense.
- For clarification, you may refer to the file ```../cayley_codes.md``` for information about where this project is going, and the context of things we're implementing.

### Step 1
- inside code/core.py, add the following:
- For the ring $R=F_2[F_2]$, add a class defining a map between free modules $f:R^n\rightarrow R^m$.
The class should consist of (1) an overall support matching $F_2$ elements with integers $[0,\ldots,s-1]$, and (2) a $sm\times n$ binary matrix describing the coefficients of the linear map given the overall support. In other words, the same support applies to every column of $f$. For the binary matrix and operations, use the classes and functions provided in ```code/z2_helpers.py``` and ```code/bitgauss_wrappers.py```.
- Add a function to compose two maps of the above form.
- Add a function to compute the syzygies. I.e. input is a map $f$ and a (small) integer $l_{\max}$ and output is map $g$ of independent generators such that $fg=0$ and the image of $g$ is equal to the kernel of $f$ restricted to a support which is the depth-$l_{\max}$ star. You can do this by computing the kernel restricted to the $l$-star, increasing $l$ from $0$ to $l_{\max}$ and modding out shifted versions of the smaller-$l$ kernels, as described in ```cayley_codes.md```.

### Step 2
- add a function ```hermitian_transpose``` which returns the transpose $f^\dagger$ of a free-module map $f$ as an $R$-valued matrix, with all $R$-entries inverted.
- add a function ```generate_infinite_code``` that takes as input a number $n$ of qubits per vertex, a number $m<n$ of initial generators, and maximum syzygy star/support size $l_{\max}, and a maximum initial star size $l_{init}$ and stabilizer weight $w_{init}$.
It then generates $m$ random words of hamming weight $\leq w_{init}$ on the $l_{init}$-star, and assembles them into a free-module map $f$.
Then we return $H_X=\ker(f^\dagger)$ and $H_Z=\ker(H_X^\dagger)$, where $\ker$ means the syzygies.
It should then automatically hold that $H_X=\ker(H_Z^\dagger)$ (related to the fact that the single and triple annihilator are the same).
The syzygy should correspond to the operators of other type ($X$ vs $Z$) that commute.