using TensorAlgebra, Test

const K = Float64
const V = VectorSpace(:V,K)
const W = VectorSpace(:W,K)

const v = Vector(V,[1,2,3])
const w = Vector(W,[4,5,6])

const α = Covector(V,[1,2,3])
const β = Covector(W,[4,5,6])

@testset "Inclusion" begin
    @test (v in V) === true
    @test (v in W) === false
    @test (v in V^*) === false
    @test (v in W^*) === false
    @test (w in V) === false
    @test (w in W) === true
    @test (w in V^*) === false
    @test (w in W^*) === false
    @test (α in V) === false
    @test (α in W) === false
    @test (α in V^*) === true
    @test (α in W^*) === false
    @test (β in V) === false
    @test (β in W) === false
    @test (β in V^*) === false
    @test (β in W^*) === true
end

@testset "Product spaces" begin
    @test V×V === ProductSpace(V,V)
    @test V×W === ProductSpace(V,W)
    @test V×(V^*) === ProductSpace(V,dual(V))
    @test V×(W^*) === ProductSpace(V,dual(W))
end
