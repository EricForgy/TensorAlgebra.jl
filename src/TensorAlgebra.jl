module TensorAlgebra

using LinearAlgebra

export Tensor, Covector, VectorSpace, ProductSpace, kind, label, field, rank, dual, domain, ×

abstract type AbstractSpace{K,S,L} end

struct VectorSpace{K,S,L} <: AbstractSpace{K,S,L} end

struct ProductSpace{K,R,S,L} <: AbstractSpace{K,S,L} end

struct Tensor{K,R,D <: AbstractSpace{K}} <: AbstractArray{K,R}
    array::Array{K,R}
end

const Covector{K,L} = Tensor{K,1,VectorSpace{K,Vector{K},L}}

kind(::AbstractSpace{K,S}) where {K,S} = S

label(::AbstractSpace{K,S,L}) where {K,S,L} = L

field(::Vector{K}) where {K} = K

field(::AbstractSpace{K}) where {K} = K

field(::Tensor{K}) where {K} = K

rank(::Tensor{K,R}) where {K,R} = R

dual(::VectorSpace{K,Vector{K},L}) where {K,L} = VectorSpace{K,Covector{K,L},L}()

dual(::VectorSpace{K,Covector{K,L},L}) where {K,L} = VectorSpace{K,Vector{K},L}()

dual(ts::ProductSpace) = ProductSpace(dual.(kind(ts))...)
# dual(ts::ProductSpace) = ProductSpace(reverse(dual.(kind(ts)))...) # Consider reversing?

Base.size(t::Tensor) = size(t.array)

Base.getindex(t::Tensor,ix) = getindex(t.array, ix)

Covector(vs::VectorSpace{K},a) where {K} = Covector{K,label(vs)}(a)

Vector(vs::VectorSpace{K},a) where {K} = Tensor{K,1,VectorSpace{K,Covector{K,label(vs)},label(vs)}}(a)

VectorSpace(L::Symbol,::Type{K}) where {K} = VectorSpace{K,Vector{K},L}()

ProductSpace(args::VectorSpace{K}...) where {K} = 
    ProductSpace{K,length(args),(args...,),Symbol(join(args," × "))}()

# Tensor(vs::ProductSpace{K<R},a) where {K} = 

domain(::Tensor{K,1,D}) where {K,D} = D()

domain(::Tensor{K,R,D}) where {K,R,D} = D
    
function (f::Tensor{K,R})(x::Tensor{K,R}) where {K,R}
    dual(domain(f)) === domain(x) || error("Domain mismatch")
    dot(f.array, x.array)
end

Base.:^(v::AbstractSpace,::typeof(*)) = dual(v)

Base.show(io::IO, vs::AbstractSpace) = print(io, label(vs))

Base.show(io::IO, vs::VectorSpace{K,<:Covector{K}}) where {K} = print(io, label(vs), "^*")

Base.first(ts::ProductSpace) = first(kind(ts))

Base.last(ts::ProductSpace) = last(kind(ts))

Base.in(t::Tensor{K,1}, vs::VectorSpace{K}) where {K} = dual(domain(t)) === vs

×(vs1::VectorSpace{K},vs2::VectorSpace{K}) where {K} = ProductSpace(vs1,vs2)

×(vs::VectorSpace{K},ts::ProductSpace{K}) where {K} = ProductSpace(vs,kind(ts)...)

×(ts::ProductSpace{K},vs::VectorSpace{K}) where {K} = ProductSpace(kind(ts)...,vs)

×(ts1::ProductSpace{K},ts2::ProductSpace{K}) where {K} = ProductSpace((kind(ts1)..., kind(ts2)...))

# ⊗(t1::Tensor{K,R1},t2::Tensor{K,R2}) where {K,R1,R2} = 

end # module
