using DeRhamOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, BlockArrays, LinearAlgebra, HarmonicOrthogonalPolynomials, ArrayLayouts, Test

function harmonicpart(f)
    c = transform(Laurent(), θ -> f[RadialCoordinate(1,θ)])
    splat((x,y) -> sum(c[k] * (isodd(k) ? (x+im*y)^(k÷2) : (x-im*y)^(k÷2))  for k in colsupport(c)))
end

@testset "DiskNormal general a" begin
    𝐱 = (x, y) = SVector(0.1, 0.2)
    r, θ = norm(𝐱), atan(reverse(𝐱)...)

    @testset "weight evaluation" begin
        for a in 0:3
            W = DiskVectorWeight((a,a+1))
            @test W[𝐱] ≈ (1 - x^2 - y^2)^a * @SMatrix([1-y^2 x*y; x*y 1-x^2])
        end
    end

    @testset "equivariance: ρᵀ W^(a) ρ = (1-r²)^a diag(1, 1-r²)" begin
        for a in 0:3, r_val in [0.1, 0.5, 0.9], θ_val in range(0, 2π, length=9)[1:8]
            𝐲 = SVector(r_val*cos(θ_val), r_val*sin(θ_val))
            ρ = [cos(θ_val) -sin(θ_val); sin(θ_val) cos(θ_val)]
            lhs = ρ' * DiskVectorWeight((a,a+1))[𝐲] * ρ
            rhs = (1 - r_val^2)^a * Diagonal([1.0, 1 - r_val^2])
            @test lhs ≈ rhs atol=1e-12
        end
    end

    @testset "m=0 evaluation" begin
        for a in 0:3, J in 1:4, k in 1:2
            P = DiskVector(0, (a,a+1))
            v = P[𝐱, Block(J)[k]]
            er_vec = SVector(cos(θ), sin(θ))
            eθ_vec = SVector(-sin(θ), cos(θ))
            if k == 1
                @test v ≈ jacobip(J-1, a, 1, 2r^2-1) * r * er_vec
            else
                @test v ≈ jacobip(J-1, a+1, 1, 2r^2-1) * r * eθ_vec
            end
        end
    end

    @testset "m=0 orthogonality w.r.t. W^(a)" begin
        for a in 0:3
            P = DiskVector(0, (a, a+1))
            G = [sum(
                    P[𝐳, Block(J1)[k1]]' * DiskVectorWeight((a,a+1))[𝐳] * P[𝐳, Block(J2)[k2]]
                 for 𝐳 in UnitDisk())
                 for J1 in 1:5, k1 in 1:2, J2 in 1:5, k2 in 1:2]
            G = reshape(G, 10, 10)
            @test isapproxdiagonal(G)
            @test all(>(1e-10), diag(G))
        end
    end

    @testset "m≠0 orthogonality w.r.t. W^(a)" begin
        for a in 0:2, m in [-2, -1, 1, 2]
            P = DiskVector(m, (a,a+1))
            G = [sum(P[𝐳, j1]' * DiskVectorWeight((a,a+1))[𝐳] * P[𝐳, j2] for 𝐳 in UnitDisk())
                 for j1 in 1:8, j2 in 1:8]
            @test isapproxdiagonal(G)
            @test all(>(1e-10), real.(diag(G)))
        end
    end

    @testset "cross-mode orthogonality: m1 ≠ m2" begin
        for a in 0:2, m1 in [-2, 0, 1], m2 in [-1, 0, 2]
            m1 == m2 && continue
            P1 = DiskVector(m1, (a,a+1))
            P2 = DiskVector(m2, (a,a+1))
            G = [sum(P1[𝐳, j1]' * DiskVectorWeight((a,a+1))[𝐳] * P2[𝐳, j2] for 𝐳 in UnitDisk())
                 for j1 in 1:6, j2 in 1:6]
            @test norm(G) < 1e-10
        end
    end

    @testset "n-polynomials: Weighted(DiskNormal(m,a)) = W^(a) v" begin
        for a in 0:2, m in -1:1
            P = DiskVector(m, (a,a+1))
            W = Weighted(P)
            for J in 1:4, k in 1:min(J,2)
                @test W[𝐱, Block(J)[k]] ≈ DiskVectorWeight((a,a+1))[𝐱] * P[𝐱, Block(J)[k]]
            end
        end
    end

    @testset "a = -1" begin
        @testset "DiskBubble" begin
            w = DiskZernikeWeight(-1)
            @test w[𝐱] ≈ (1-norm(𝐱)^2)^(-1)
            ip = function(f,g)
                f̃,g̃ = harmonicpart(f),harmonicpart(g)
                sum(conj(f[𝐱] - f̃(𝐱)) * w[𝐱] * (g[𝐱] - g̃(𝐱)) for 𝐱 in UnitDisk()) + sum(conj(f[RadialCoordinate(1,θ)])g[RadialCoordinate(1,θ)] for θ in 0..2π)
            end
            for m = -3:3
                Q = DiskBasis(m, -1)
                @test isapproxdiagonal([ip(Q[:,k], Q[:,j]) for k =1:5, j=1:5])
            end
        end

        @testset "DiskNormal" begin
            𝐱 = x,y = SVector(0.1,0.2)
            for m = -3:3
                W = DiskVectorWeight((0,-1))
                @test W[𝐱] ≈ @SMatrix([1-x^2 -x*y; -x*y 1-y^2])/(1-x^2-y^2)

                𝐏 = DiskVector(m)
                𝐐 = DiskVector(m, (0,-1))
                R = 𝐏\𝐐

                @test R[1:5,1:5] ≈ (𝐏'𝐏)[1:5,1:5] \ [sum(𝐏[𝐱,k]'𝐐[𝐱,j] for 𝐱 in UnitDisk()) for k=1:5,j=1:5]

                @test (𝐐'𝐐)[1:5,1:5] ≈ [sum(𝐐[𝐱,k]'𝐐[𝐱,j] for 𝐱 in UnitDisk()) for k=1:5, j=1:5]

                # ip = function(f,g)
                #     f̃,g̃ = harmonicpart(f),harmonicpart(g)
                #     sum(conj(f[𝐱] - f̃(𝐱)) * W[𝐱] * (g[𝐱] - g̃(𝐱)) for 𝐱 in UnitDisk()) + sum(conj(f[RadialCoordinate(1,θ)])g[RadialCoordinate(1,θ)] for θ in 0..2π)
                # end

                # f = 𝐐[:,1]
                # ip(𝐐[:,1], 𝐐[:,1])
            end
        end
        
        @testset "grad" begin
            𝐱 = SVector(0.1,0.2)
            for a = -1:5, m = -5:5
                Q = DiskBasis(m, a)
                ∇ = grad(Q)
                for n = 1:10
                    @test gradient(𝐱 -> DiskBasis{eltype(𝐱)}(m,a)[𝐱,n], 𝐱) ≈ ∇[𝐱,n]
                end
                if a == -1
                    @test (∇'∇)[1:10,1:10] ≈ [sum(∇[𝐱,k]'∇[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10, j=1:10] 
                end
            end
        end

        @testset "curl" begin
            𝐱 = SVector(0.1,0.2)
            for a = -1:3, m = -3:3
                𝐏 = DiskVector(m, (a+1,a))
                curl𝐏 = curl(𝐏)
                for k = 2:6
                    @test curl𝐏[𝐱,k] ≈ fd_curl(𝐱 -> DiskVector{eltype(𝐱)}(m, (a+1,a))[𝐱,k], 𝐱)
                end
            end
        end
    end
end

# n = 3
# gradient(𝐱 -> DiskBasis{eltype(𝐱)}(m,a)[𝐱,n], 𝐱)
# r = norm(𝐱)
# jacobip(n+1, a, 0, 2r^2-1)
# DiskVector(0, (a+1,a))[𝐱, Block(n-1)[1]] * (Jacobi(a+1,1) \ diff(Jacobi(a,0)))[n-1,n]