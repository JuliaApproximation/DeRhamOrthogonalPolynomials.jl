using DeRhamOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, BlockArrays, Test


@testset "PeriodicCylinder" begin
    @testset "scalar" begin
        (x,y,z) = 𝐱 = SVector(0.1,0.2,0.3)
        for m = -2:2, ω = -2:2
            P = PeriodicCylinderBasis(m, ω)
            Q = PeriodicCylinderBubble(m, ω)
            W = Weighted(Q)

            @test P[𝐱, 1:5] ≈ DiskBasis(m)[SVector(x,y),1:5] * exp(im*ω*z)
            @test W[𝐱, 1:5] ≈ Weighted(DiskBasis(m,1))[SVector(x,y),1:5] * exp(im*ω*z)
            @test (P\W)[1:11,1:10]' * P[𝐱,1:11] ≈ W[𝐱,1:10]
            @test (Q\P)[1:11,1:10]' * Q[𝐱,1:11] ≈ P[𝐱,1:10]
            @test (P'P)[1:4,1:4] ≈ [sum(sum(P[SVector(x,y,z),k]'P[SVector(x,y,z),j] for (x,y) in UnitDisk()) for z in 0..2π) for k=1:4, j=1:4]
            @test (W'W)[1:4,1:4] ≈ [sum(sum(W[SVector(x,y,z),k]'W[SVector(x,y,z),j] for (x,y) in UnitDisk()) for z in 0..2π) for k=1:4, j=1:4]
        end
    end

    @testset "vector" begin
        (x,y,z) = 𝐱 = SVector(0.1,0.2,0.3)
        for m=setdiff(-1:2,0), ω = -2:1
            P = PeriodicCylinderVector(m, ω)
            @test P[𝐱,1] ≈ only(P[𝐱,Block(1)]) ≈ P[𝐱,Block(1)[1]] ≈ [DiskVector(m)[SVector(x,y),1]; 0]*exp(im*ω*z)
            @test P[𝐱,2] ≈ only(P[𝐱,Block(2)]) ≈ P[𝐱,Block(2)[1]] ≈ [0,0,DiskBasis(m)[SVector(x,y),1]]*exp(im*ω*z)
            @test P[𝐱,3:4] ≈ P[𝐱,Block(3)] ≈ [P[𝐱,Block(3)[1]],P[𝐱,Block(3)[2]]] ≈ [[DiskVector(m)[SVector(x,y),2]; 0]*exp(im*ω*z),[DiskVector(m)[SVector(x,y),3]; 0]*exp(im*ω*z)]
            @test (P'P)[1:4,1:4] ≈ [sum(sum(P[SVector(x,y,z),k]'P[SVector(x,y,z),j] for (x,y) in UnitDisk()) for z in 0..2π) for k=1:4, j=1:4]
        end
    end

    @testset "normal" begin
        (x,y,z) = 𝐱 = SVector(0.1,0.2,0.3)
        for m=setdiff(-1:2,0), ω = -2:1
            P = PeriodicCylinderVector(m, ω)
            Q = PeriodicCylinderNormal(m, ω)
            W = Weighted(Q)
            @test Q[𝐱,1] ≈ only(Q[𝐱,Block(1)]) ≈ Q[𝐱,Block(1)[1]] ≈ [DiskNormal(m)[SVector(x,y),1]; 0]*exp(im*ω*z)
            @test Q[𝐱,2] ≈ only(Q[𝐱,Block(2)]) ≈ Q[𝐱,Block(2)[1]] ≈ [0,0,DiskBasis(m,1)[SVector(x,y),1]]*exp(im*ω*z)
            @test Q[𝐱,3:4] ≈ Q[𝐱,Block(3)] ≈ [Q[𝐱,Block(3)[1]],Q[𝐱,Block(3)[2]]] ≈ [[DiskNormal(m)[SVector(x,y),2]; 0]*exp(im*ω*z),[DiskNormal(m)[SVector(x,y),3]; 0]*exp(im*ω*z)]
            @test W[𝐱,1] ≈ only(W[𝐱,Block(1)]) ≈ W[𝐱,Block(1)[1]] ≈ [Weighted(DiskNormal(m))[SVector(x,y),1]; 0]*exp(im*ω*z)
            @test W[𝐱,2] ≈ only(W[𝐱,Block(2)]) ≈ W[𝐱,Block(2)[1]] ≈ [0,0,Weighted(DiskBasis(m,1))[SVector(x,y),1]]*exp(im*ω*z)

            @test (Q \ P)[1:10,1:10]'Q[𝐱,1:10] ≈ P[𝐱,1:10]
            @test (P \ W)[1:13,1:10]'P[𝐱,1:13] ≈ W[𝐱,1:10]

            @test (W'W)[1:4,1:4] ≈ [sum(sum(W[SVector(x,y,z),k]'W[SVector(x,y,z),j] for (x,y) in UnitDisk()) for z in 0..2π) for k=1:4, j=1:4]
        end
    end
end