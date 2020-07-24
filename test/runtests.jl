using TensorAlgebra, Test

function setup(::Type{K}=Float64) where {K}
    U = VectorSpace(:U,K,dims=(2,))
    V = VectorSpace(:V,K,dims=(3,))
    W = VectorSpace(:W,K,dims=(4,))

    u = Vector(U,[1,2])
    v = Vector(V,[1,2,3])
    w = Vector(W,[1,2,3,4])

    α = Covector(U,[1,2])
    β = Covector(V,[1,2,3])
    γ = Covector(W,[1,2,3,4])

    tuv = Tensor(U⊗V,[1 2 3;4 5 6])
    tvv = Tensor(V⊗V,[1 2 3;4 5 6;7 8 9])
    tvw = Tensor(V⊗W,[1 2 3 4;5 6 7 8;9 10 11 12])
    auvw = Array{Float64,3}(undef,2,3,4)
    auvw[:,:,1] = [1 2 3;4 5 6]
    auvw[:,:,2] = [7 8 9;10 11 12]
    auvw[:,:,3] = [13 14 15;16 17 18]
    auvw[:,:,4] = [19 20 21;22 23 24]
    tuvw = Tensor(U⊗V⊗W,auvw)

    uα = u⊗α
    uvw = u⊗v⊗w
    αβγ = α⊗β⊗γ

    U,V,W,u,v,w,α,β,γ,tuv,tvv,tvw,tuvw,uα,uvw,αβγ
end

U,V,W,u,v,w,α,β,γ,tuv,tvv,tvw,tuvw,uα,uvw,αβγ = setup(Float64)

@testset "Dimension errors" begin
    @test_throws DimensionMismatch Vector(V,[1,2])
    @test_throws DimensionMismatch Vector(V,[1,2,3,4])
end

@testset "Domain errors" begin
    @test_throws DomainError α(v)
    @test_throws DomainError (α⊗β)(-,u)
    @test_throws DomainError (α⊗β)(v,-)
    @test_throws DomainError α⊗β+β⊗α
    @test_throws DomainError α⊗β-β⊗α
end

@testset "Inclusion" begin
    @test (v ∈ U) === false
    @test (v ∈ V) === true
    @test (v ∈ W) === false
    @test (v ∈ U^*) === false
    @test (v ∈ V^*) === false
    @test (v ∈ W^*) === false
    @test (w ∈ U) === false
    @test (w ∈ V) === false
    @test (w ∈ W) === true
    @test (w ∈ U^*) === false
    @test (w ∈ V^*) === false
    @test (w ∈ W^*) === false
    @test (α ∈ U) === false
    @test (α ∈ V) === false
    @test (α ∈ W) === false
    @test (α ∈ U^*) === true
    @test (α ∈ V^*) === false
    @test (α ∈ W^*) === false
    @test (β ∈ U) === false
    @test (β ∈ V) === false
    @test (β ∈ W) === false
    @test (β ∈ U^*) === false
    @test (β ∈ V^*) === true
    @test (β ∈ W^*) === false
    @test (α⊗β ∈ (U⊗V)^*) === true
end

@testset "Product spaces" begin
    @test V×V === ProductSpace(V,V)
    @test V×W === ProductSpace(V,W)
    @test V×(V^*) === ProductSpace(V,dual(V))
    @test V×(W^*) === ProductSpace(V,dual(W))
    @test V⊗V === TensorSpace(V,V)
    @test V⊗W === TensorSpace(V,W)
    @test V⊗(V^*) === TensorSpace(V,dual(V))
    @test V⊗(W^*) === TensorSpace(V,dual(W))
end

@testset "Addition, subtraction and negation" begin
    @test u+u ≈ 2*u
    @test u-u ≈ 0*u
    @test -u ≈ -1*u
end

@testset "Tensor products" begin
    @test (domain(uα) === U⊗U) === false
    @test (domain(uα) === (U^*)⊗U) === true
    @test (domain(uα) === U⊗U^*) === false
    @test (domain(uα) === (U⊗U)^*) === false
    @test (domain(uvw) === U⊗V⊗W) === false
    @test (domain(uvw) === (U^*)⊗V⊗W) === false
    @test (domain(uvw) === U⊗(V^*)⊗W) === false
    @test (domain(uvw) === U⊗V⊗W^*) === false
    @test (domain(uvw) === ((U⊗V)^*)⊗W) === false
    @test (domain(uvw) === U⊗(V⊗W)^*) === false
    @test (domain(uvw) === (U⊗V⊗W)^*) === true
    @test (domain(αβγ) === U⊗V⊗W) === true
end

@testset "Evaluation" begin
    @test α(u) ≈ 5.0
    @test u(α) ≈ 5.0
    @test (α⊗β)(u,v) ≈ α(u)*β(v)
    @test (α⊗β)(u⊗v) ≈ α(u)*β(v)
    @test (α⊗β)(-,v)(u) ≈ α(u)*β(v)
    @test (α⊗β)(u,-)(v) ≈ α(u)*β(v)
    @test (u⊗v)(α,β) ≈ u(α)*v(β)
    @test (u⊗v)(α⊗β) ≈ u(α)*v(β)
    @test (u⊗v)(-,β)(α) ≈ u(α)*v(β)
    @test (u⊗v)(α,-)(β) ≈ u(α)*v(β)
    @test (α⊗β⊗γ)(u,v,w) ≈ α(u)*v(β)*γ(w)
    @test (α⊗β⊗γ)(u⊗v⊗w) ≈ α(u)*v(β)*γ(w)
    @test (α⊗β⊗γ)(-,v,w)(u) ≈ α(u)*v(β)*γ(w)
    @test (α⊗β⊗γ)(u,-,w)(v) ≈ α(u)*v(β)*γ(w)
    @test (α⊗β⊗γ)(u,v,-)(w) ≈ α(u)*v(β)*γ(w)
    @test (α⊗β⊗γ)(-,-,w)(u,v) ≈ α(u)*v(β)*γ(w)
    @test (α⊗β⊗γ)(-,v,-)(u,w) ≈ α(u)*v(β)*γ(w)
    @test (α⊗β⊗γ)(u,-,-)(v,w) ≈ α(u)*v(β)*γ(w)
    @test tuv(-,v)(u) ≈ tuv(u,v)
    @test tuv(u,-)(v) ≈ tuv(u,v)
    @test tuvw(-,v,w)(u) ≈ tuvw(u,v,w)
    @test tuvw(u,-,w)(v) ≈ tuvw(u,v,w)
    @test tuvw(u,v,-)(w) ≈ tuvw(u,v,w)
    @test tuvw(-,-,w)(u,v) ≈ tuvw(u,v,w)
    @test tuvw(-,v,-)(u,w) ≈ tuvw(u,v,w)
    @test tuvw(u,-,-)(v,w) ≈ tuvw(u,v,w)
    @test (α⊗β⊗γ⊗tuvw)(u,v,w,u,v,w) ≈ α(u)*v(β)*γ(w)*tuvw(u,v,w)
end

