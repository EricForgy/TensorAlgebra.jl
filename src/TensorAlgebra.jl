module TensorAlgebra

using LinearAlgebra

export Tensor, Covector, VectorSpace, TensorSpace, dual, field, space, symbol

abstract type AbstractSpace{T,L} end

struct VectorSpace{T,S,L} <: AbstractSpace{T,L} end

struct Tensor{T,N,D <: VectorSpace} <: AbstractArray{T,N}
    array::Array{T,N}
end

Base.size(t::Tensor) = size(t.array)

Base.getindex(t::Tensor,ix) = getindex(t.array, ix)

(t::Tensor{T,N,D})(a::D) where {T,N,D} = dot(t.array, a)

const Covector{T,S} = Tensor{T,1,VectorSpace{Vector{T},S}}

Covector(S::Symbol,a::Vector{T}) where {T} = Covector{T,S}(a)

Covector(a::Vector{T}) where {T} = Covector{T,:V}(a)

Vector(S::Symbol,a::Array{T,1}) where {T} = Tensor{T,1,VectorSpace{T,Covector{T},S}}(a)

(v::Vector{T})(c::Covector{T}) where {T} = dot(v, c.array)

(c::Covector{T})(v::Vector{T}) where {T} = dot(c.array, v)

dual(::VectorSpace{T,Vector{T},L}) where {T,L} = VectorSpace{T,Covector{T},L}()

dual(::VectorSpace{T,Covector{T},L}) where {T,L} = VectorSpace{T,Vector{T},L}()

function (f::Tensor{T,N,D})(x::Tensor{T,N,dualD}) where {T,N,D,dualD}
    # dual(D()) === dualD() || error("Domain mismatch")
    dot(f.array, x.array)
end

field(::Type{Vector{T}}) where {T} = T

field(::Type{Tensor{T}}) where {T} = T

field(::VectorSpace{T,S}) where {T,S} = field(S)

space(::VectorSpace{T,S}) where {T,S} = S

symbol(::VectorSpace{T,S,L}) where {T,S,L} = L

symbol(::Tensor{T,N,AbstractSpace{T,L}}) where {T,N,L} = L

VectorSpace(L::Symbol,::Type{S}) where {S} = VectorSpace{field(S),S,L}()

VectorSpace(::Type{S}) where {S} = VectorSpace{field(S),S,:V}()

Base.:^(v::VectorSpace,::typeof(*)) = dual(v)

Base.show(io::IO, vs::VectorSpace{T,Vector{T}}) where {T} = print(io, symbol(vs))

Base.show(io::IO, vs::VectorSpace{T,<:Covector{T}}) where {T} = print(io, symbol(vs), "^*")

struct TensorSpace{T,N,L} <: AbstractSpace{T,L}
    spaces::NTuple{N,VectorSpace{T}}
end

TensorSpace(args::VectorSpace{T}...) where {T} = 
    TensorSpace{T,length(args),Symbol(join(string.(args)," x "))}((args...,))

dual(ts::TensorSpace) = TensorSpace(reverse(dual.(ts.spaces))...)

symbol(ts::TensorSpace) = join(symbol.(ts.spaces), " x ")

Base.first(ts::TensorSpace) = first(ts.spaces)

Base.last(ts::TensorSpace) = last(ts.spaces)

Base.:*(vs1::VectorSpace{T},vs2::VectorSpace{T}) where {T} = TensorSpace(vs1,vs2)

Base.:*(vs::VectorSpace{T},ts::TensorSpace{T}) where {T} = TensorSpace(vs,ts.spaces...)

Base.:*(ts::TensorSpace{T},vs::VectorSpace{T}) where {T} = TensorSpace(ts.spaces...,vs)

Base.:*(ts1::TensorSpace{T},ts2::TensorSpace{T}) where {T} = TensorSpace((ts1.spaces..., ts2.spaces...))

Base.:^(v::TensorSpace,::typeof(*)) = dual(v)

Base.show(io::IO, ts::TensorSpace) = print(io, join(ts.spaces, " ⨉ "))

Base.show(io::IO, ts::Type{<:TensorSpace}) = print(io, join(ts().spaces, " ⨉ "))

end # module
