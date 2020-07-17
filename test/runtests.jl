using TensorAlgebra, Test

function setup()
    K = Float64
    V = VectorSpace(:V,Covector{K})
    W = VectorSpace(:W,Covector{K})
    V,W
end
        
# @testset "Basic currencies" begin
#     for (pos, s, u, c, n) in currencies
#         ccy = currency(pos)
#         @test symbol(ccy) == s
#         @test unit(ccy) == u
#         @test name(ccy) == n
#         @test code(ccy) == c
#     end
# end
    
# @testset "All currencies" begin
#     for sym in Currencies.allsymbols()
#         ccy = Currency{sym}
#         ct = cash(sym)
#         @test ct == cash(ccy)
#         @test ct == cash(sym)
#         @test currency(ct) == ccy
#         @test symbol(ct) == symbol(ccy)
#         @test unit(ct) == unit(ccy)
#         @test code(ct) == code(ccy)
#         @test name(ct) == name(ccy)

#         pos = Position(ct,1)
#         pt = typeof(pos)
#         @test currency(pt) == currency(ct)
#         @test currency(pt) == ccy
#         @test 1pt == pos
#         @test pos * 1 == pos
#         @test 1pos + 1pos == Position(ct,2)
#         @test 1pos - 1pos == Position(ct,0)
#         @test 20pos / 4pos == FixedDecimal{Int,unit(ct)}(5)
#         @test 20pos / 4 == Position(ct,5)
#     end
# end
