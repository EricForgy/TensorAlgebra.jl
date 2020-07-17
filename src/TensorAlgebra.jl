module TensorAlgebra

using LinearAlgebra

export Tensor, Covector, VectorSpace, ProductSpace, kind, label, field, rank, dual, domain, ×

abstract type AbstractSpace{K,R,S,L} end

struct VectorSpace{K,S,L} <: AbstractSpace{K,1,S,L} end

struct ProductSpace{K,R,S,L} <: AbstractSpace{K,R,S,L} end

struct Tensor{K,R,D <: AbstractSpace{K}} <: AbstractArray{K,R}
    array::Array{K,R}
end

const Covector{K,L} = Tensor{K,1,VectorSpace{K,Vector{K},L}}

field(::AbstractSpace{K}) where {K} = K

field(::Tensor{K}) where {K} = K

field(::Vector{K}) where {K} = K

rank(::AbstractSpace{K,R}) where {K,R} = R

rank(::Tensor{K,R}) where {K,R} = R

kind(::AbstractSpace{K,R,S}) where {K,R,S} = S

label(::AbstractSpace{K,R,S,L}) where {K,R,S,L} = L

dual(::VectorSpace{K,Vector{K},L}) where {K,L} = VectorSpace{K,Covector{K,L},L}()

dual(::VectorSpace{K,Covector{K,L},L}) where {K,L} = VectorSpace{K,Vector{K},L}()

dual(ts::ProductSpace) = ProductSpace(dual.(kind(ts))...)
# dual(ts::ProductSpace) = ProductSpace(reverse(dual.(kind(ts)))...) # Consider reversing?

Covector(vs::VectorSpace{K},a) where {K} = Covector{K,label(vs)}(a)

Vector(vs::VectorSpace{K},a) where {K} = Tensor{K,1,VectorSpace{K,Covector{K,label(vs)},label(vs)}}(a)

VectorSpace(L::Symbol,::Type{K}) where {K} = VectorSpace{K,Vector{K},L}()

ProductSpace(args::VectorSpace{K}...) where {K} = 
    ProductSpace{K,length(args),(args...,),Symbol(join(args," × "))}()

Tensor(::P,a::Array{K,R}) where {K,R,P<:ProductSpace{K,R}} = Tensor{K,R,P}(a)

domain(::Tensor{K,R,D}) where {K,R,D} = D()

# domain(::Tensor{K,R,D}) where {K,R,D} = D
    
function (f::Tensor{K,R})(x::Tensor{K,R}) where {K,R}
    dual(domain(f)) === domain(x) || error("Domain mismatch")
    dot(f.array, x.array)
end

Base.size(t::Tensor) = size(t.array)

Base.getindex(t::Tensor,ix::Vararg{Int}) = getindex(t.array, ix...)

Base.isassigned(t::Tensor, ix::Vararg{Int}) = isassigned(t.array, ix...)

Base.:^(v::AbstractSpace,::typeof(*)) = dual(v)

Base.first(ts::ProductSpace) = first(kind(ts))

Base.last(ts::ProductSpace) = last(kind(ts))

Base.in(t::Tensor{K,1}, vs::VectorSpace{K}) where {K} = dual(domain(t)) === vs

×(vs1::VectorSpace{K},vs2::VectorSpace{K}) where {K} = ProductSpace(vs1,vs2)

×(vs::VectorSpace{K},ts::ProductSpace{K}) where {K} = ProductSpace(vs,kind(ts)...)

×(ts::ProductSpace{K},vs::VectorSpace{K}) where {K} = ProductSpace(kind(ts)...,vs)

×(ts1::ProductSpace{K},ts2::ProductSpace{K}) where {K} = ProductSpace((kind(ts1)..., kind(ts2)...))

# ⊗(t1::Tensor{K,R1},t2::Tensor{K,R2}) where {K,R1,R2} = 

Base.show(io::IO, vs::AbstractSpace) = print(io, label(vs))

Base.show(io::IO, vs::VectorSpace{K,<:Covector{K}}) where {K} = print(io, label(vs), "⃰")

Base.show(io::IO, vs::Type{<:AbstractSpace}) = print(io, label(vs()))

end # module
