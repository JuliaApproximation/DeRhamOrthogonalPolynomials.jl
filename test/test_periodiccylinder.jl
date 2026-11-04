using DeRhamOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, BlockArrays, Test


@testset "PeriodicCylinder" begin
    @testset "scalar" begin
        (x,y,z) = 𝐱 = SVector(0.1,0.2,0.3)
        for m = -2:2, ω = -2:2
            P = PeriodicCylinderBasis(m, ω)
            Q = PeriodicCylinderBasis(m, ω, 1)
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
        for m=-1:2, ω = -2:1
            P = PeriodicCylinderVector(m, ω)
            if iszero(m) # DiskVector(0) has blocks of size 2
                @test P[𝐱,1:2] ≈ P[𝐱,Block(1)] ≈ [P[𝐱,Block(1)[1]],P[𝐱,Block(1)[2]]] ≈ [[DiskVector(m)[SVector(x,y),1]; 0]*exp(im*ω*z),[DiskVector(m)[SVector(x,y),2]; 0]*exp(im*ω*z)]
                @test P[𝐱,3] ≈ only(P[𝐱,Block(2)]) ≈ P[𝐱,Block(2)[1]] ≈ [0,0,DiskBasis(m)[SVector(x,y),1]]*exp(im*ω*z)
                @test P[𝐱,4:5] ≈ P[𝐱,Block(3)] ≈ [P[𝐱,Block(3)[1]],P[𝐱,Block(3)[2]]] ≈ [[DiskVector(m)[SVector(x,y),3]; 0]*exp(im*ω*z),[DiskVector(m)[SVector(x,y),4]; 0]*exp(im*ω*z)]
            else
                @test P[𝐱,1] ≈ only(P[𝐱,Block(1)]) ≈ P[𝐱,Block(1)[1]] ≈ [DiskVector(m)[SVector(x,y),1]; 0]*exp(im*ω*z)
                @test P[𝐱,2] ≈ only(P[𝐱,Block(2)]) ≈ P[𝐱,Block(2)[1]] ≈ [0,0,DiskBasis(m)[SVector(x,y),1]]*exp(im*ω*z)
                @test P[𝐱,3:4] ≈ P[𝐱,Block(3)] ≈ [P[𝐱,Block(3)[1]],P[𝐱,Block(3)[2]]] ≈ [[DiskVector(m)[SVector(x,y),2]; 0]*exp(im*ω*z),[DiskVector(m)[SVector(x,y),3]; 0]*exp(im*ω*z)]
            end
            @test (P'P)[1:4,1:4] ≈ [sum(sum(P[SVector(x,y,z),k]'P[SVector(x,y,z),j] for (x,y) in UnitDisk()) for z in 0..2π) for k=1:4, j=1:4]
        end
    end

    @testset "normal" begin
        (x,y,z) = 𝐱 = SVector(0.1,0.2,0.3)
        for m=-1:2, ω = -2:1
            P = PeriodicCylinderVector(m, ω)
            Q = PeriodicCylinderVector(m, ω, (1,0,1))
            W = Weighted(Q)
            if iszero(m)
                @test Q[𝐱,1:2] ≈ Q[𝐱,Block(1)] ≈ [Q[𝐱,Block(1)[1]],Q[𝐱,Block(1)[2]]] ≈ [[DiskVector(m,(0,1))[SVector(x,y),1]; 0],[DiskVector(m,(0,1))[SVector(x,y),2]; 0]]*exp(im*ω*z)
                @test Q[𝐱,3] ≈ only(Q[𝐱,Block(2)]) ≈ Q[𝐱,Block(2)[1]] ≈ [0,0,DiskBasis(m,1)[SVector(x,y),1]]*exp(im*ω*z)
                @test W[𝐱,1:2] ≈ W[𝐱,Block(1)] ≈ [W[𝐱,Block(1)[1]],W[𝐱,Block(1)[2]]] ≈ [[Weighted(DiskVector(m,(0,1)))[SVector(x,y),1]; 0],[Weighted(DiskVector(m,(0,1)))[SVector(x,y),2]; 0]]*exp(im*ω*z)
                @test W[𝐱,3] ≈ only(W[𝐱,Block(2)]) ≈ W[𝐱,Block(2)[1]] ≈ [0,0,Weighted(DiskBasis(m,1))[SVector(x,y),1]]*exp(im*ω*z)
            else
                @test Q[𝐱,1] ≈ only(Q[𝐱,Block(1)]) ≈ Q[𝐱,Block(1)[1]] ≈ [DiskVector(m,(0,1))[SVector(x,y),1]; 0]*exp(im*ω*z)
                @test Q[𝐱,2] ≈ only(Q[𝐱,Block(2)]) ≈ Q[𝐱,Block(2)[1]] ≈ [0,0,DiskBasis(m,1)[SVector(x,y),1]]*exp(im*ω*z)
                @test Q[𝐱,3:4] ≈ Q[𝐱,Block(3)] ≈ [Q[𝐱,Block(3)[1]],Q[𝐱,Block(3)[2]]] ≈ [[DiskVector(m,(0,1))[SVector(x,y),2]; 0]*exp(im*ω*z),[DiskVector(m,(0,1))[SVector(x,y),3]; 0]*exp(im*ω*z)]
                @test W[𝐱,1] ≈ only(W[𝐱,Block(1)]) ≈ W[𝐱,Block(1)[1]] ≈ [Weighted(DiskVector(m,(0,1)))[SVector(x,y),1]; 0]*exp(im*ω*z)
                @test W[𝐱,2] ≈ only(W[𝐱,Block(2)]) ≈ W[𝐱,Block(2)[1]] ≈ [0,0,Weighted(DiskBasis(m,1))[SVector(x,y),1]]*exp(im*ω*z)
            end

            @test (Q \ P)[1:10,1:10]'Q[𝐱,1:10] ≈ P[𝐱,1:10]
            @test (P \ W)[1:13,1:10]'P[𝐱,1:13] ≈ W[𝐱,1:10]

            @test (W'W)[1:4,1:4] ≈ [sum(sum(W[SVector(x,y,z),k]'W[SVector(x,y,z),j] for (x,y) in UnitDisk()) for z in 0..2π) for k=1:4, j=1:4]
        end
    end

    @testset "grad" begin
        𝐱 = SVector(0.1,0.2,0.3)
        for m = -1:2, ω = -2:3
            Q = PeriodicCylinderBasis(m, ω, -1)
            ∇ = grad(Q)
            for n = 1:10
                @test gradient(𝐱 -> PeriodicCylinderBasis{eltype(𝐱)}(m, ω, -1)[𝐱,n], 𝐱) ≈ ∇[𝐱,n]
            end
            @test (∇'∇)[1:3,1:3] ≈ [sum(sum(∇[[𝐱;z],k]'∇[[𝐱;z],j] for 𝐱 in UnitDisk()) for z in 0..2π) for k=1:3, j=1:3]
        end
    end
end