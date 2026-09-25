# Implementation plan

These notes contain instructions for Claude to implement the ideas in cayley_codes.md step by step.

### Step 1
- Create a file code/core.py.
- For the ring $R=F_2[F_2]$, add a class defining a map between free modules $f:R^n\rightarrow R^m$.
The class should consist of (1) an overall support matching $F_2$ elements with integers $[0,\ldots,s-1]$, and (2) a $sm\times n$ binary matrix describing the coefficients of the linear map given the overall support. In other words, the same support applies to every column of $f$. For the binary matrix and operations, use the classes and functions provided in ```code/z2_helpers.py``` and ```code/bitgauss_wrappers.py```.
- Add a function to compose two maps of the above form.
- Add a function to compute the syzygies. I.e. input is a map $f$ and a (small) integer $l_{\max}$ and output is map $g$ of independent generators such that $fg=0$ and the image of $g$ is equal to the kernel of $f$ restricted to a support which is the depth-$l_{\max}$ star. You can do this by computing the kernel restricted to the $l$-star, increasing $l$ from $0$ to $l_{\max}$ and modding out shifted versions of the smaller-$l$ kernels, as described in ```cayley_codes.md```.