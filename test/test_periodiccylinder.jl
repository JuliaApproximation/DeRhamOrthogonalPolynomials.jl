using MultivariateOrthogonalPolynomials, DeRhamOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, BlockArrays, LinearAlgebra, ContinuumArrays, Test
using ContinuumArrays: basis, coefficients


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

    @testset "grad m = $m, ω = $ω" for m = -1:2, ω = -2:3
        𝐱 = SVector(0.1,0.2,0.3)
        Q = PeriodicCylinderBasis(m, ω, -1)
        ∇ = grad(Q)
        for n = 1:10
            @test gradient(𝐱 -> PeriodicCylinderBasis{eltype(𝐱)}(m, ω, -1)[𝐱,n], 𝐱) ≈ ∇[𝐱,n]
        end
        @test (∇'∇)[1:3,1:3] ≈ [sum(sum(∇[[𝐱;z],k]'∇[[𝐱;z],j] for 𝐱 in UnitDisk()) for z in 0..2π) for k=1:3, j=1:3]
    end

    @testset "curl" begin
        𝐱 = SVector(0.1,0.2,0.3)
        for m = -1:2, ω = -2:3
            V = PeriodicCylinderVector(m, ω, (-1,0,-1))
            C = curl(V)
            for n = 1:10
                @test fd_curl(𝐱 -> PeriodicCylinderVector{eltype(𝐱)}(m, ω, (-1,0,-1))[𝐱,n], 𝐱) ≈ C[𝐱,n] atol=1E-10
            end
            ∇ = grad(PeriodicCylinderBasis(m, ω, -1))
            @test norm((coefficients(C) * (V \ ∇))[1:20,1:10]) ≤ 1E-10 # curl ∘ grad == 0
            @test (C'C)[1:3,1:3] ≈ [sum(sum(C[[𝐱;z],k]'C[[𝐱;z],j] for 𝐱 in UnitDisk()) for z in 0..2π) for k=1:3, j=1:3]
        end
        @test_throws AssertionError curl(PeriodicCylinderVector(1, 1)) # 𝐮 must be in DiskVector(m, (a+1,a))
    end

    @testset "divergence" begin
        𝐱 = SVector(0.1,0.2,0.3)
        for m = -1:2, ω = -2:3
            W = PeriodicCylinderVector(m, ω, (0,-1,0))
            D = divergence(W)
            for n = 1:10
                @test fd_div(𝐱 -> PeriodicCylinderVector{eltype(𝐱)}(m, ω, (0,-1,0))[𝐱,n], 𝐱) ≈ D[𝐱,n] atol=1E-10
            end
            C = curl(PeriodicCylinderVector(m, ω, (-1,0,-1)))
            @test basis(C) == W
            @test norm((coefficients(D) * coefficients(C))[1:20,1:10]) ≤ 1E-10 # divergence ∘ curl == 0
            @test (D'D)[1:3,1:3] ≈ [sum(sum(D[[𝐱;z],k]'D[[𝐱;z],j] for 𝐱 in UnitDisk()) for z in 0..2π) for k=1:3, j=1:3]
        end
        @test_throws AssertionError divergence(PeriodicCylinderVector(1, 1)) # 𝐮 must be in DiskVector(m, (a,a+1))
    end

    @testset "expansions" begin
        @testset "grad" begin
            f = ((x,y), z) -> exp(cos(z)sin(x-y^2+1))
            M = 25 # total modes
            Z,L = ComplexZernike(),Laurent()
            gˣʸ, plˣʸ = ContinuumArrays.plan_grid_transform(Z, Block(M))
            gᶻ, plᶻ = ContinuumArrays.plan_grid_transform(L, Block(M))
            F = f.(gˣʸ, reshape(gᶻ, 1, 1, :))
            F̂ = similar(F, ComplexF64)
            for ℓ in axes(F,3)
                F̂[:,:,ℓ] = (plˣʸ*F[:,:,ℓ]).matrix
            end
            for k in axes(F,1), j in axes(F,2)
                F̂[k,j,:] = plᶻ * F̂[k,j,:]
            end

            evalcylinder(F̂, 𝐱, a=0) = sum(sum((m = (-1)^isodd(j) * (j ÷ 2);
                                          ω = (-1)^isodd(ℓ) * (ℓ ÷ 2);
                                          kr = 1:(size(F̂,1) - (abs(m)+3)÷2); # "triangle" of coefficients
                                          fₘ = PeriodicCylinderBasis(m,ω,a)[:,kr]F̂[kr,j,ℓ];
                                          fₘ[𝐱]) for j in axes(F̂, 2)) for ℓ in axes(F̂,3))

            function lowercylinder!(F̂, a=-1)
                for ℓ in axes(F̂, 3), j in axes(F̂, 2)
                    m = (-1)^isodd(j) * (j ÷ 2)
                    ω = (-1)^isodd(ℓ) * (ℓ ÷ 2)
                    kr = 1:(size(F̂,1) - (abs(m)+3)÷2)
                    R = ((PeriodicCylinderBasis(m,ω) \ PeriodicCylinderBasis(m,ω,a)))[kr,kr]
                    ldiv!(R, view(F̂, kr,j,ℓ))
                end
                F̂
            end
            lowercylinder(F̂, a...) = lowercylinder!(copy(F̂), a...)

            F₋ = lowercylinder(F̂)

            
            evalgradcylinder(F̂, 𝐱, a=0) = sum(sum((m = (-1)^isodd(j) * (j ÷ 2);
            ω = (-1)^isodd(ℓ) * (ℓ ÷ 2);
            kr = 1:(size(F̂,1) - (abs(m)+3)÷2); # "triangle" of coefficients
            fₘ = PeriodicCylinderBasis(m,ω,a)[:,kr]F̂[kr,j,ℓ];
            grad(fₘ)[𝐱]) for j in axes(F̂, 2)) for ℓ in axes(F̂,3))
            
            let 𝐱 = (x,y,z) = SVector(0.1,0.2,0.3)
                @test evalcylinder(F̂, 𝐱) ≈ evalcylinder(F₋, 𝐱, -1) ≈ f((x,y), z)
                @test evalgradcylinder(F₋, 𝐱, -1) ≈ evalgradcylinder(F̂, 𝐱) ≈ [exp(cos(z)sin(x-y^2+1))cos(x-y^2+1)cos(z),-2y*exp(cos(z)sin(x-y^2+1))cos(x-y^2+1)cos(z),-exp(cos(z)sin(x-y^2+1))sin(z)sin(x-y^2+1)]
            end
        end  
        
        @testset "curl" begin
            # Transforms not implemented yet
            for m = -3:3, ω = -3:3
                c = 1:8
                fₘ = PeriodicCylinderVector(m,ω, (-1,0,-1)) * [c; zeros(∞)]
                let 𝐱 = SVector(0.1,0.2,0.3)
                    @test curl(fₘ)[𝐱] ≈ fd_curl(𝐲 -> (PeriodicCylinderVector{eltype(𝐲)}(m, ω, (-1,0,-1))[:,axes(c,1)] * c)[𝐲], 𝐱)
                end
            end
        end

         @testset "divergence" begin
            # Transforms not implemented yet
            for m = -3:3, ω = -3:3
                c = 1:8
                fₘ = PeriodicCylinderVector(m,ω, (0,-1,0)) * [c; zeros(∞)]
                let 𝐱 = SVector(0.1,0.2,0.3)
                    @test divergence(fₘ)[𝐱] ≈ fd_div(𝐲 -> (PeriodicCylinderVector{eltype(𝐲)}(m, ω, (0,-1,0))[:,axes(c,1)] * c)[𝐲], 𝐱)
                end
            end
        end
    end
end