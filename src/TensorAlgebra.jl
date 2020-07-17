module TensorAlgebra

using LinearAlgebra

export Tensor, Covector, VectorSpace, ProductSpace, dual, field, kind, label

abstract type AbstractSpace{K,S,L} end

kind(::AbstractSpace{K,S}) where {K,S} = S

label(::AbstractSpace{K,S,L}) where {K,S,L} = L

struct VectorSpace{K,S,L} <: AbstractSpace{K,S,L} end

struct ProductSpace{K,R,S,L} <: AbstractSpace{K,S,L} end

struct Tensor{K,R,D <: AbstractSpace} <: AbstractArray{K,R}
    array::Array{K,R}
end

const Covector{K,L} = Tensor{K,1,VectorSpace{K,Vector{K},L}}

function field end

field(::Type{Vector{K}}) where {K} = K

field(::Type{<:AbstractSpace{K}}) where {K} = K

field(::Type{<:Tensor{K}}) where {K} = K

rank(::Tensor{K,R}) where {K,R} = R

domain(::Tensor{K,R,D}) where {K,R,D} = D

Base.size(t::Tensor) = size(t.array)

Base.getindex(t::Tensor,ix) = getindex(t.array, ix)

Covector(L::Symbol,a::Vector{K}) where {K} = Covector{K,L}(a)

Vector(L::Symbol,a::Array{K,1}) where {K} = Tensor{K,1,VectorSpace{K,Covector{K},L}}(a)

VectorSpace(L::Symbol,::Type{S}) where {S} = VectorSpace{field(S),S,L}()

VectorSpace(::Type{S}) where {S} = VectorSpace{field(S),S,:V}()

ProductSpace(args::VectorSpace{K}...) where {K} = 
    ProductSpace{K,length(args),(args...,),Symbol(join(string.(args)," x "))}()

dual(::VectorSpace{K,Vector{K},L}) where {K,L} = VectorSpace{K,Covector{K},L}()

dual(::VectorSpace{K,Covector{K},L}) where {K,L} = VectorSpace{K,Vector{K},L}()

dual(ts::ProductSpace) = ProductSpace(dual.(kind(ts))...)
# dual(ts::ProductSpace) = ProductSpace(reverse(dual.(kind(ts)))...)

function (f::Tensor{K,R,D})(x::Tensor{K,R,dualD}) where {K,R,D,dualD}
    dual(D) === dualD || error("Domain mismatch")
    dot(f.array, x.array)
end

Base.:^(v::VectorSpace,::typeof(*)) = dual(v)

Base.show(io::IO, vs::VectorSpace{K,Vector{K}}) where {K} = print(io, label(vs))

Base.show(io::IO, vs::VectorSpace{K,<:Covector{K}}) where {K} = print(io, label(vs), "^*")

Base.first(ts::ProductSpace) = first(kind(ts))

# Base.last(ts::ProductSpace) = last(ts.spaces)

# Base.:*(vs1::VectorSpace{K},vs2::VectorSpace{K}) where {K} = ProductSpace(vs1,vs2)

# Base.:*(vs::VectorSpace{K},ts::ProductSpace{K}) where {K} = ProductSpace(vs,ts.spaces...)

# Base.:*(ts::ProductSpace{K},vs::VectorSpace{K}) where {K} = ProductSpace(ts.spaces...,vs)

# Base.:*(ts1::ProductSpace{K},ts2::ProductSpace{K}) where {K} = ProductSpace((ts1.spaces..., ts2.spaces...))

# Base.:^(v::ProductSpace,::typeof(*)) = dual(v)

# Base.show(io::IO, ts::ProductSpace) = print(io, join(ts.spaces, " ⨉ "))

# Base.show(io::IO, ts::Type{<:ProductSpace}) = print(io, join(ts().spaces, " ⨉ "))

end # module
