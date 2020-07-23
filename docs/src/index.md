# TensorAlgebra.jl

This package attempts to implement tensor algebra in a natural Julian way that stays as true as possible to the underlying mathematics.

---

## Background and Motivation

There are several other tensor packages in the Julia ecosystem such as:

- [TensorKit.jl](https://jutho.github.io/TensorKit.jl/stable/): July 28, 2014, Commits: >340
  - [TensorOperations.jl](https://jutho.github.io/TensorOperations.jl/stable/): March 31, 2014, Commits: >207
    - Used by TensorKit.jl.
- [Tensors.jl](https://kristofferc.github.io/Tensors.jl/stable/): Feb 22, 2016, Commits: >244
- [ITensors.jl](https://itensor.github.io/ITensors.jl/stable/): January 18, 2019, Commits: >1789
- [CoolTensors.jl](https://simeonschaub.github.io/CoolTensors.jl/dev/): July 14, 2020, Commits: >19

with related packages:

- [Catlab.jl](https://epatters.github.io/Catlab.jl/dev/): January 23, 2017, Commits: >1170
- [TensorCore.jl](https://epatters.github.io/Catlab.jl/dev/): May 5, 2020, Commits: >2

We briefly say a few words about each package (not thorough reviews) that help to motivate this package.

### TensorKit.jl

Of the tensor packages considered, TensorKit.jl has been around the longest with its initial commit on July 28, 2014. TensorKit.jl is a mature package that hits on the key topics that a general tensor package should consider, e.g.

- Vector spaces
- Dual spaces
- Product spaces
- Tensor products
- Etc.

It does so with a view to category theory giving it a strong mathematical foundation.

The package defines a custom type, `TensorMap`, representing multilinear maps:

```math
F: W_1\otimes\cdots\otimes W_{N_2}\to V_1\otimes\cdots\otimes V_{N_1}.
```

with a special case when ``N_1 = 0`` called `Tensor` so that a tensor is a multilinear map

```math
t: W_1\otimes\cdots\otimes W_{N_2} \to K
```

where ``K`` is the underlying field.

One issue with this is that the same tensor ``t\in V^*\otimes W^*`` can be thought of as a linear map in multiple ways, e.g.

```math
t: V\otimes W \to K,\\
```

with ``t(v,w) \in K`` for ``v\in V`` and ``w\in W``. However, we also have

```math
t: W \to V^*,
```

with ``t(-,w) \in V^*`` and

```math
t: V \to W^*,
```

with ``t(v,-) \in W^*``. In other words, with partial evaluation - common for tensor algebra - the same tensor can be thought of as a `TensorMap` in multiple inequalivalent ways.

### Tensors.jl

From the README:

> This Julia package provides fast operations with symmetric and non-symmetric tensors of order 1, 2 and 4.

Also from the README:

> Supports Automatic Differentiation to easily compute first and second order derivatives of tensorial functions.

With the restriction to tensors of order 1,2 and 4, Tensors.jl is not intended to be a general purpose tensor package, but rather a highly-optimized package that appears to have arisen from real-world applications. It also does not distinguish contravariant and covariant tensors, which would be desriable for a general purpose tensor package.

### ITensors.jl

From the README:

> ITensors is a library for rapidly creating correct and efficient tensor network algorithms.

Although much newer than both TensorKit.jl and Tensors.jl with an initial commit on January 18, 2019, ITensors.jl has already received a lot of attention with over 1700 commits.

With a focus on tensor networks and applications in quantum computing, among other things, ITensors.jl seems specialized and does not serve as a general purpose tensor algebra package, e.g. like Tensors.jl, it does not distinguish contravariant and covariant tensors.

### CoolTensors.jl

CoolTensors.jl is a new package that introduces a string literal to indicate whether a tensor coeeficient index should be "up" (covariant) or "down" (contravariant). For example, a contravariant tensor is constructed via

```julia
X = T"'"[1, 2, 3]
```

and a covariant tensor is constructed via

```julia
α = T","[1, 2, 3]
```

Once defined, tensors components are accessed like usual arrays, e.g. `X[1]` and `α[2]`.

It does not yet support other elementary concepts such as vector space, dual spaces, etc. (although impled), but it does build upon TensorCore.jl (see below).

### Catlab.jl

From the README:

>Catlab.jl is a framework for applied and computational category theory, written in the Julia language. Catlab provides a programming library and interactive interface for applications of category theory to scientific and engineering fields. It emphasizes monoidal categories due to their wide applicability but can support any categorical structure that is formalizable as a generalized algebraic theory.

Although much more general than a tensor package, it is worth noting that Catlab.jl is a mature package that continues with active and rapid development. A natural way to think of both linear algebra and tensor algebra is as the study of the category ``Vect_k`` whose objects are vector spaces and whose morphisms are (multi)linear maps.

### TensorCore.jl

TensorCore.jl is a simple package that came to life as a result of an extended discussion in a PR to the standard library: `LinearAlgebra`.

- [Support `⊙` and `⊗` as elementwise- and outer-product operators](https://github.com/JuliaLang/julia/pull/35150)

### TensorAlgebra.jl

The same discussion that led to the creation of TensorCore.jl also inspired the issue:

- [LinearAlgebra and TensorCore future design: Be more greedy](https://github.com/JuliaLang/julia/issues/35763)

That issue and the discussions therein as well as discussions on Slack and Zulip motivated the development of this package.

Although the standard library, `LinearAlgebra`, is already quite good, there remains room for improvement to help facilitate better tensor packages such as this one and the ones listed above. It is hoped that this package can help facilitate that discussion.

---

## Introduction

Of the packages mentioned about, TensorAlgebra.jl shares the most in common with TensorKit.jl. In this section, we briefly go over the main concepts.

### Vector Spaces

Linear algebra and, by extension, tensor algebra are both about vector spaces and maps between them. As such, vector spaces are fundamental concepts and should have a first-class role.

In TensorAlgebra.jl, a vector space is constructed via

```julia
julia> V = VectorSpace(:V,Float64)
V
```

A vector is then an element of a vector space and can be constructed via

```julia
julia> v = Vector(V,[1,2,3])
3-element Tensor{Float64,1,V^*}:
 1.0
 2.0
 3.0
```

 with

```julia
julia> v ∈ V
true
```

There are two things to note above:

 - Arrays are promoted to the type of the respective vector space
 - The type of a vector is `Tensor{Float64,1,V^*}`

The third parameter of the vector type, i.e. `V^*`, denotes the dual space of the original vector space `V`.

Like TensorKit.jl, a tensor is thought of conceptually as a multilinear map

```math
t: V \to K
```

from a vector space ``V`` to the underlying field ``K``.

In this package, we identify the dual of a dual space with the original vector space, i.e.

```math
V^{**} \simeq V
```

since there is a natural transformation

```math
\hat{v}(\alpha) := \alpha(v)
```

for ``\hat{v}\in V^{**}``, ``\alpha\in V^*`` and ``v\in V``.

In this way, a vector is a tensor whose domain is the dual space ``V^*``, i.e.

```math
v: V^* \to K.
```

Similarly, a covector ``\alpha\in V^*`` is a tensor whose domain is the vector space ``V``, i.e.

```math
\alpha: V \to K.
```

A covector can be constructed via

```julia
julia> α = Covector(V,[1,2,3])
3-element Tensor{Float64,1,V}:
 1.0
 2.0
 3.0
```

with

```julia
julia> α ∈ V^*
true
``` 

```julia
julia> α(v)
14.0
```

and

```julia
julia> v(α)
14.0
```

### Tensor Spaces

Just as a vector is an element of a vector space, a tensor is an element of a tensor space. Given vector spaces ``V`` and ``W``, we can construct a tensor space ``V\otimes W`` via

```julia
julia> TensorSpace(V,W)
V ⊗ W
```

or using the unicode `⊗` (obtained from the REPL using `\otimes[tab]`)

```julia
julia> V⊗W
V ⊗ W
```

Recall, a tensor ``t\in V\otimes W`` is a map

```math
t: V^* \otimes W^* \to K
```

so we can construct the tensor ``t`` via

```julia
julia> t = Tensor((V⊗W)^*,[1 2 3 4;5 6 7 8;9 10 11 12])
3×4 Tensor{Float64,2,V^* ⊗ W^*}:
 1.0   2.0   3.0   4.0
 5.0   6.0   7.0   8.0
 9.0  10.0  11.0  12.0
```

with 

```julia
julia> t ∈ V⊗W
true
```

Now, with

```julia
julia> α
3-element Tensor{Float64,1,V}:
 1.0
 2.0
 3.0
```

from above, let

```julia
julia> β = Covector(W,[1,2,3,4])
4-element Tensor{Float64,1,W}:
 1.0
 2.0
 3.0
 4.0
```

and consider

```julia
julia> α⊗β
3×4 TensorProduct{Float64,2,Tuple{Tensor{Float64,1,V},Tensor{Float64,1,W}}}:
 1.0  2.0  3.0   4.0
 2.0  4.0  6.0   8.0
 3.0  6.0  9.0  12.0
```

with

```julia
julia> α⊗β ∈ (V⊗W)^*
true
```

We can now evaluate

```julia
julia> t(α⊗β)
500.0
```

### Partial Evaluation

In addition to considering a tensor to be a linear map

```math
t: V_1\otimes\cdots\otimes V_N\to K
```

we can also consider it a multilinear function

```math
t: V_1 \times\cdots\times V_N\to K
```

such that

```julia
julia> t(α,β)
500.0
```

In this way, we can consider partial evaluation:

```julia
julia> t(-,β)
3-element Tensor{Float64,1,V^*}:
  30.0
  70.0
 110.0
```

and

```julia
julia> t(α,-)
4-element Tensor{Float64,1,W^*}:
 38.0
 44.0
 50.0
 56.0
```

with

```julia
julia> t(-,β) ∈ V
true

julia> t(α,-) ∈ W
true
```

such that

```julia
julia> t(α,β) === t(-,β)(α) === t(α,-)(β)
true
```

### Indexing

Tensors coefficients are often presented in texts with upper and lower indices to keep track of contravariant and covariant components, respectively. However, an advantage of a typed programming language such as Julia is that we can embed this information into the instance itself so, for example, a vector `v` knows it is an element of a vector space `V`. Hence, there is no need for us to consider alternative indexing methods and we can rely on the usual `Array` indices.

For example

```julia
julia> t[2,3]
7.0
```

and

```julia
julia> (α⊗β)[2,3]
6.0
```

## Acknowledgements

I would like to acknowledge with appreciation the constructive discussions in the PR

- [Support `⊙` and `⊗` as elementwise- and outer-product operators](https://github.com/JuliaLang/julia/pull/35150)

and the issue

- [LinearAlgebra and TensorCore future design: Be more greedy](https://github.com/JuliaLang/julia/issues/35763)

as well as various conversations that took place on both Slack and Zulip.