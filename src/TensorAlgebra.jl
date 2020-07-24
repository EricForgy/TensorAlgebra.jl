module TensorAlgebra

using LinearAlgebra

export Tensor, Covector, VectorSpace, DualSpace, ProductSpace, TensorSpace, TensorProduct
export spaces, tensors, label, dimensions, field, degree, dual, domain, ×, ⊗

abstract type AbstractSpace{K,N} end

struct VectorSpace{K,L,D} <: AbstractSpace{K,1} end

struct DualSpace{K,L,D} <: AbstractSpace{K,1} end

abstract type AbstractProductSpace{K,N,S} <: AbstractSpace{K,N} end

struct ProductSpace{K,N,S} <: AbstractProductSpace{K,N,S} end

struct TensorSpace{K,N,S} <: AbstractProductSpace{K,N,S} end

abstract type AbstractTensor{K,N} <: AbstractArray{K,N} end

struct Tensor{K,N,D} <: AbstractTensor{K,N}
    array::Array{K,N}
end

const Covector{K,L,D} = Tensor{K,1,VectorSpace{K,L,D}}

struct TensorProduct{K,N,S} <: AbstractTensor{K,N}
    scalar::K
    tensors::S
end

Base.collect(t::Tensor) = t.array

field(::AbstractSpace{K}) where {K} = K

field(::AbstractTensor{K}) where {K} = K

field(::Vector{K}) where {K} = K

degree(::AbstractSpace{K,N}) where {K,N} = N

degree(::AbstractTensor{K,N}) where {K,N} = N

spaces(as::AbstractSpace{K,1}) where {K} = (as,)

spaces(::AbstractProductSpace{K,N,S}) where {K,N,S} = S

label(::Union{VectorSpace{K,L},DualSpace{K,L}}) where {K,L} = L

dimensions(::Union{VectorSpace{K,L,D},DualSpace{K,L,D}}) where {K,L,D} = D

dual(::VectorSpace{K,L,D}) where {K,L,D} = DualSpace{K,L,D}()

dual(::DualSpace{K,L,D}) where {K,L,D} = VectorSpace{K,L,D}()

function Covector(vs::VectorSpace{K,L,D}, a) where {K,L,D}
    size(a)[1] === D || throw(DimensionMismatch("Tried to create a covector of dimension $(size(a)) in the vector space $(label(vs)) of dimension $(D)."))
    Tensor{K,1,TensorSpace{K,1,(VectorSpace{K,L,D}(),)}}(a)
end

function Vector(vs::VectorSpace{K,L,D}, a) where {K,L,D}
    size(a)[1] === D || throw(DimensionMismatch("Tried to create a vector of dimension $(size(a)) in the vector space $(label(vs)) of dimension $(D)."))
    Tensor{K,1,TensorSpace{K,1,(DualSpace{K,L,D}(),)}}(a)
end

VectorSpace(L::Symbol,::Type{K},dim::Int) where {K,D} = VectorSpace{K,L,dim}()

ProductSpace(args::AbstractSpace{K,1}...) where {K} = ProductSpace{K,length(args),(args...,)}()

TensorSpace(args::Vararg{<:AbstractSpace{K,1}}) where {K} = TensorSpace{K,length(args),(args...,)}()

TensorSpace(::ProductSpace{K,N,S}) where {K,N,S} = TensorSpace{K,N,S}()

TensorSpace(ts::TensorSpace{K,N,S}) where {K,N,S} = ts

(::Type{PS})(args...) where {PS <: ProductSpace} = ProductSpace(args...)

(::Type{TS})(args...) where {TS <: TensorSpace} = TensorSpace(args...)

dual(ps::PS) where {PS <: AbstractProductSpace} = PS(dual.(spaces(ps))...)

Tensor(dom::D,a) where {K,N,D <: AbstractSpace{K,N}} = Tensor{K,N,typeof(TensorSpace(dom))}(a)

Tensor(::D,a) where {K,N,D <: TensorSpace{K,N}} = Tensor{K,N,D}(a)

TensorProduct(args::Tensor{K}...) where {K} = TensorProduct{K,sum(degree.(args)),typeof((args...,))}(one(K), (args...,))

TensorProduct(t::Tensor) = t

domain(::Tensor{K,N,D}) where {K,N,D} = D()

domain(tp::TensorProduct{K,N,S}) where {K,N,S} = TensorSpace{K,N,((spaces.(domain.(tensors(tp)))...)...,)}()

dimensions(at::AbstractTensor) = ((dimensions.(spaces(domain(at)))...)...,)

scalar(::Tensor{K}) where {K} = one(K)

scalar(tp::TensorProduct) = tp.scalar

tensors(t::Tensor) = t

tensors(tp::TensorProduct) = tp.tensors

function (f::AbstractTensor{K,2})(::typeof(-), x::Tensor{K,1}) where {K}
    dom = spaces(domain(f))
    codom = spaces(domain(x))
    dual(dom[2]) === codom[1] || throw(DomainError(x,"Domain mismatch: Expected $(dual(dom[2])). Got $(codom[1])."))
    Tensor(dom[1], collect(f) * collect(x))
end

function (f::AbstractTensor{K,2})(x::Tensor{K,1}, ::typeof(-)) where {K}
    dom = spaces(domain(f)) 
    codom = spaces(domain(x))
    dual(dom[1]) === codom[1] || throw(DomainError(x,"Domain mismatch: Expected $(dual(dom[1])). Got $(codom[1])."))
    Tensor(dom[2], (collect(x)' * collect(f)).parent)
end

function (f::AbstractTensor{K,N})(x::AbstractTensor{K,N}) where {K,N}
    dual(domain(f)) === domain(x) || throw(DomainError(x,"Domain mismatch: Expected $(dual(domain(f))). Got $(domain(x))."))
    dot(f, x)
end

(f::Tensor{K,N})(::Vararg{typeof(-),N}) where {K,N} = f

function (f::Tensor{K,1})(x::Tensor{K,1}) where {K}
    dual(domain(f)) === domain(x) || throw(DomainError(x,"Domain mismatch: Expected $(dual(domain(f))). Got $(domain(x))."))
    dot(f, x)
end

function (f::Tensor{K,N})(xs::Vararg{Tensor{K,1},N}) where {K,N}
    x = TensorProduct(xs...)
    dual(domain(f)) === domain(x) || throw(DomainError(x,"Domain mismatch: Expected $(dual(domain(f))). Got $(domain(x))."))
    dot(f, x)
end

function (f::AbstractTensor{K,N})(xs::Vararg{Union{typeof(-),AbstractTensor{K,1}},N}) where {K,N}
    isarg = xs .!== -
    indims = findall(isarg)
    x = TensorProduct(xs[indims]...)
    any(.!isarg) || return dot(f, x)
    outdims = findall(.!isarg)
    a = reshape(mapslices(f, dims=indims) do slice
        dot(x, slice)
    end,size(f)[outdims]...)
    d = TensorSpace(spaces(domain(f))[outdims]...)
    Tensor(d, a)
end

Base.size(t::Tensor) = size(t.array)

Base.size(t::TensorProduct) = ((size.(t.tensors)...)...,)

Base.getindex(t::Tensor,ix::Vararg{Int}) = getindex(t.array, ix...)

function Base.getindex(tp::TensorProduct, ix::Vararg{Int})
    value = one(field(tp))
    offset = 0
    for (i, tensor) in enumerate(tp.tensors)
        it = degree(tensor) === 1 ? ix[offset + 1] : ix[offset + 1:offset + degree(tensor)]
        value *= getindex(tensor, it...)
        offset += degree(tensor)
    end
    value * scalar(tp)
end

Base.:^(v::AbstractSpace,::typeof(*)) = dual(v)

Base.first(ts::AbstractProductSpace) = first(spaces(ts))

Base.last(ts::AbstractProductSpace) = last(spaces(ts))

Base.in(t::AbstractTensor{K}, vs::AbstractSpace{K}) where {K} = dual(domain(t)) === TensorSpace(vs)

function Base.:+(at1::AbstractTensor{K,N}, at2::AbstractTensor{K,N}) where {K,N}
    domain(at1) === domain(at2) || throw(DomainError(at2,"Domain mismatch: Expected $(domain(at1)). Got $(domain(at2))."))
    Tensor{K,N,typeof(domain(at1))}(collect(at1) + collect(at2))
end

function Base.:-(at1::AbstractTensor{K,N}, at2::AbstractTensor{K,N}) where {K,N}
    domain(at1) === domain(at2) || throw(DomainError(at2,"Domain mismatch: Expected $(domain(at1)). Got $(domain(at2))."))
    Tensor{K,N,typeof(domain(at1))}(collect(at1) - collect(at2))
end

function Base.:-(t::Tensor{K,N,D}) where {K,N,D}
    Tensor{K,N,D}(-t.array)
end

function Base.:-(tp::TensorProduct{K,N,S}) where {K,N,S}
    TensorProduct{K,N,S}(-tp.scalar, tp.tensors)
end

Base.:*(x::Number,t::Tensor{K,N,D}) where {K,N,D} = Tensor{K,N,D}(x * t.array)

Base.:*(t::Tensor{K,N,D},x::Number) where {K,N,D} = Tensor{K,N,D}(x * t.array)

Base.:*(x::Number,tp::TensorProduct{K,N,S}) where {K,N,S} = TensorProduct{K,N,S}(x * scalar(tp), tensors(tp))

Base.:*(tp::TensorProduct{K,N,S},x::Number) where {K,N,S} = x * tp

×(vs1::AbstractSpace{K,1},vs2::AbstractSpace{K,1}) where {K} = ProductSpace(vs1, vs2)

×(vs::AbstractSpace{K,1},ts::ProductSpace{K}) where {K} = ProductSpace(vs, spaces(ts)...)

×(ts::ProductSpace{K},vs::AbstractSpace{K,1}) where {K} = ProductSpace(spaces(ts)..., vs)

×(ts1::ProductSpace{K},ts2::ProductSpace{K}) where {K} = ProductSpace((spaces(ts1)..., spaces(ts2)...))

⊗(vs1::AbstractSpace{K,1},vs2::AbstractSpace{K,1}) where {K} = TensorSpace(vs1, vs2)

⊗(vs::AbstractSpace{K,1},ts::TensorSpace{K}) where {K} = TensorSpace(vs, spaces(ts)...)

⊗(ts::TensorSpace{K},vs::AbstractSpace{K,1}) where {K} = TensorSpace(spaces(ts)..., vs)

⊗(ts1::TensorSpace{K},ts2::TensorSpace{K}) where {K} = TensorSpace((spaces(ts1)..., spaces(ts2)...))

⊗(t1::Tensor{K,R1},t2::Tensor{K,R2}) where {K,R1,R2} = TensorProduct{K,R1 + R2,typeof((t1, t2))}(one(K), (t1, t2))

⊗(tp::TensorProduct{K,R1},t::Tensor{K,R2}) where {K,R1,R2} = TensorProduct{K,R1 + R2,typeof((tensors(tp)..., t))}(scalar(tp), (tensors(tp)..., t))

⊗(t::Tensor{K,R1},tp::TensorProduct{K,R2}) where {K,R1,R2} = TensorProduct{K,R1 + R2,typeof((t, tensors(tp)...))}(scalar(tp), (t, tensors(tp)...))

⊗(t1::TensorProduct{K,R1},t2::TensorProduct{K,R2}) where {K,R1,R2} = TensorProduct{K,R1 + R2,typeof((tensors(t1)..., tensors(t2)...))}(scalar(t1) * scalar(t2), (tensors(t1)..., tensors(t2)...))

⊗(t1::AbstractTensor,t2::AbstractTensor,ts::Vararg{<:AbstractTensor}) = TensorProduct(scalar(t1) * scalar(t2), t1 ⊗ t2, ts...)

Base.show(io::IO, ::VectorSpace{K,L}) where {K,L} = print(io, L)

# Base.show(io::IO, ::Type{VectorSpace{K,L}}) where {K,L} = print(io, L)

Base.show(io::IO, ::DualSpace{K,L}) where {K,L} = print(io, L, "^*")

# Base.show(io::IO, ::Type{DualSpace{K,L}}) where {K,L} = print(io, L, "^*")

# Base.show(io::IO, ::ProductSpace{K,N,S}) where {K,N,S} = print(io, join(S, " × "))

# Base.show(io::IO, ::Type{ProductSpace{K,N,S}}) where {K,N,S} = print(io, join(S, " × "))

# Base.show(io::IO, ::TensorSpace{K,N,S}) where {K,N,S} = print(io, join(S, " ⊗ "))

# Base.show(io::IO, ::Type{TensorSpace{K,N,S}}) where {K,N,S} = print(io, join(S, " ⊗ "))

end # module
