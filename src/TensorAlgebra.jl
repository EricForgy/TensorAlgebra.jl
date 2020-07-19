module TensorAlgebra

using LinearAlgebra

export Tensor, Covector, VectorSpace, ProductSpace, spaces, label, field, degree, dual, domain, ×

abstract type AbstractSpace{K,N} end

struct VectorSpace{K,L} <: AbstractSpace{K,1} end

struct DualSpace{K,L} <: AbstractSpace{K,1} end

struct ProductSpace{K,N,S} <: AbstractSpace{K,N} end

struct Tensor{K,N,D <: AbstractSpace{K}} <: AbstractArray{K,N}
    array::Array{K,N}
end

const Covector{K,L} = Tensor{K,1,VectorSpace{K,L}}

field(::AbstractSpace{K}) where {K} = K

field(::Tensor{K}) where {K} = K

field(::Vector{K}) where {K} = K

degree(::AbstractSpace{K,N}) where {K,N} = N

degree(::Tensor{K,N}) where {K,N} = N

spaces(::ProductSpace{K,N,S}) where {K,N,S} = S

label(::Union{VectorSpace{K,L},DualSpace{K,L}}) where {K,L} = L

dual(::VectorSpace{K,L}) where {K,L} = DualSpace{K,L}()

dual(::DualSpace{K,L}) where {K,L} = VectorSpace{K,L}()

dual(ps::ProductSpace) = ProductSpace(dual.(spaces(ps))...)
# dual(ps::ProductSpace) = ProductSpace(reverse(dual.(spaces(ps)))...) # Consider reversing?

Covector(::VectorSpace{K,L},a) where {K,L} = Tensor{K,1,VectorSpace{K,L}}(a)

Vector(::VectorSpace{K,L},a) where {K,L} = Tensor{K,1,DualSpace{K,L}}(a)

VectorSpace(L::Symbol,::Type{K}) where {K} = VectorSpace{K,L}()

ProductSpace(args::AbstractSpace{K,1}...) where {K} = 
    ProductSpace{K,length(args),(args...,)}()

Tensor(::D,a::Array{K,N}) where {K,N,D<:AbstractSpace{K,N}} = Tensor{K,N,D}(a)

domain(::Tensor{K,1,D}) where {K,D} = D()

domain(::Tensor{K,N,D}) where {K,N,D} = D
    
function (f::Tensor{K,N})(x::Tensor{K,N}) where {K,N}
    dual(domain(f)) === domain(x) || error("Domain mismatch")
    dot(f.array, x.array)
end

Base.size(t::Tensor) = size(t.array)

Base.getindex(t::Tensor,ix::Vararg{Int}) = getindex(t.array, ix...)

Base.:^(v::AbstractSpace,::typeof(*)) = dual(v)

Base.first(ts::ProductSpace) = first(spaces(ts))

Base.last(ts::ProductSpace) = last(spaces(ts))

Base.in(t::Tensor{K,1}, vs::AbstractSpace{K,1}) where {K} = dual(domain(t)) === vs

×(vs1::AbstractSpace{K,1},vs2::AbstractSpace{K,1}) where {K} = ProductSpace(vs1,vs2)

×(vs::AbstractSpace{K,1},ts::ProductSpace{K}) where {K} = ProductSpace(vs,spaces(ts)...)

×(ts::ProductSpace{K},vs::AbstractSpace{K,1}) where {K} = ProductSpace(spaces(ts)...,vs)

×(ts1::ProductSpace{K},ts2::ProductSpace{K}) where {K} = ProductSpace((spaces(ts1)..., spaces(ts2)...))

# ⊗(t1::Tensor{K,R1},t2::Tensor{K,R2}) where {K,R1,R2} = 

Base.show(io::IO, ::VectorSpace{K,L}) where {K,L} = print(io, L)

Base.show(io::IO, ::DualSpace{K,L}) where {K,L} = print(io, L, "⃰")

Base.show(io::IO, ::ProductSpace{K,N,S}) where {K,N,S} = print(io, join(S, " × "))

end # module
