using DeRhamOrthogonalPolynomials, MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, ContinuumArrays, StaticArrays, BlockArrays, LinearAlgebra, Test
using DeRhamOrthogonalPolynomials: _complexzernikemode, _complexzernikecolumn, _diskvectordegree
using MultivariateOrthogonalPolynomials: ModalTrav

@testset "DiskVector transform" begin
    @testset "basic usage" begin
        𝐟 = splat((x,y) -> SVector(exp(x*sin(y+1)), cos(exp(x)*(y+2)-1)))
        for N = 20:25, 𝐚 in ((0,0), (-1,0), (1,0), (0,-1), (0,1))
            Pl = DiskVectorPlan(N, 𝐚)
            g = grid(Pl)
            F = Pl * 𝐟.(g)
            𝐱 = SVector(0.1,0.2)
            kr,jr = axes(F)
            @test sum((DiskVector((-1)^iseven(j) * j ÷ 2, 𝐚)[:,kr]F[:,j])[𝐱]
                for j in axes(F,2)) ≈ 𝐟(𝐱)
        end
    end

    @testset "ComplexZernike column ordering" begin
        @test [_complexzernikemode(c) for c in 1:7] == [0, -1, 1, -2, 2, -3, 3]
        @test all(_complexzernikecolumn(_complexzernikemode(c)) == c for c in 1:41)
    end

    @testset "conversion from DiskVector(m, (a,a)) to DiskVector(m, (a+1,a+1)): m=$m, a=$a" for m in -4:4, a in (-1, 0)
        n = 12
        P, Q = DiskVector(m, (a+1,a+1)), DiskVector(m, (a,a))
        R = (P \ Q)[1:n,1:n]
        for 𝐱 in (SVector(0.3, -0.2), SVector(-0.6, 0.5), SVector(0.0, 0.9))
            @test [Q[𝐱, j] for j in 1:n] ≈ [sum(R[i,j] * P[𝐱, i] for i in 1:n) for j in 1:n]
        end
    end

    weights = ((-1,-1), (0,0), (1,1), (0,1), (1,0), (0,-1), (-1,0))

    @testset "DiskVectorPlan recovers every basis field: 𝐚=$𝐚, N=$N" for 𝐚 in weights, N in (7, 8)
        P = DiskVectorPlan(N, 𝐚)
        𝐱 = grid(P)
        for m in -N:N, r in 1:2N
            _diskvectordegree(m, r) ≤ N-1 || continue
            V = DiskVector(m, 𝐚)
            G = P * V[𝐱, r]
            E = zeros(ComplexF64, size(G))
            E[r, _complexzernikecolumn(m)] = 1
            @test G ≈ E atol=1e-12
        end
    end

    @testset "a vector field of degree N-1 can have mode N" begin
        N = 5
        P = DiskVectorPlan(N)
        G = P * [(x[1] + im*x[2])^(N-1) * SVector(1, im) for x in grid(P)]
        @test size(G, 2) == _complexzernikecolumn(N)
        @test G[1, end] ≈ 1
        @test norm(G) ≈ 1
    end

    @testset "DiskVectorPlan(N, a) is DiskVectorPlan(N, (a,a)): a=$a" for a in (-1, 0, 1)
        𝐟 = 𝐱 -> SVector(exp(𝐱[1])*cos(𝐱[2]), sin(𝐱[1] + 2𝐱[2]))
        P, Q = DiskVectorPlan(9, a), DiskVectorPlan(9, (a,a))
        F = 𝐟.(grid(P))
        @test P * F == Q * F
        @test inv(P) * (P * F) == inv(Q) * (Q * F)
        @test DiskVectorIPlan(9, a) * (P * F) == inv(P) * (P * F)
    end

    @testset "disk2diskvector from ComplexZernike coefficients" begin
        M = 40
        f₁ = [exp(x*cos(3*(y-1))) for (x,y) in UnitDisk()]
        f₂ = [cos(x*sin(3*(y-1))) for (x,y) in UnitDisk()]
        F₁ = ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ f₁
        F₂ = ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ f₂
        G = disk2diskvector(F₁, F₂)
        @test size(G) == (2size(F₁.matrix,1), size(F₁.matrix,2)+2)
        n = size(G,1)
        for 𝐱 in (SVector(0.1,0.2), SVector(-0.5,0.3), SVector(0.0,0.0), SVector(0.3,-0.6))
            𝐯 = sum(sum(DiskVector(_complexzernikemode(j))[𝐱,1:n] .* G[:,j]) for j in axes(G,2))
            @test 𝐯 ≈ [f₁[𝐱], f₂[𝐱]]
        end
        P = DiskVectorPlan(M+1)
        @test P * [SVector(f₁[𝐱], f₂[𝐱]) for 𝐱 in grid(P)] ≈ G
        @test_throws AssertionError disk2diskvector(F₁.matrix, F₂.matrix[:,1:end-1])
    end

    @testset "lowering Weighted(DiskVector(m, $𝐚)): m=$m" for 𝐚 in ((0,1), (1,0), (1,1)), m in (-2, 0, 1)
        V, W = DiskVector(m), Weighted(DiskVector(m, 𝐚))
        L = V \ W
        @test istril(L[1:10,1:10])
        𝐱 = SVector(0.3, -0.4)
        for k in 1:8
            @test sum(V[𝐱,j] * L[j,k] for j in 1:12) ≈ W[𝐱,k]
        end
        # no tangential component, no normal component, or neither, on the boundary
        𝐲 = SVector(cos(0.7), sin(0.7))
        for k in 1:10
            𝐚[1] == 1 && @test abs(dot(𝐲, W[𝐲,k])) ≤ 1e-13
            𝐚[2] == 1 && @test abs(dot(SVector(-𝐲[2], 𝐲[1]), W[𝐲,k])) ≤ 1e-13
        end
    end

    @testset "unsupported parameters" begin
        @test_throws ErrorException DiskVectorPlan(5, -2)
        @test_throws ErrorException DiskVectorIPlan(5, -2)
        @test_throws ErrorException DiskVectorPlan(5, (0,2))
        @test_throws ErrorException DiskVectorIPlan(5, (-2,-1))
        @test_throws ErrorException DiskVectorPlan(5, (-2,-2))
        @test_throws ErrorException DiskVectorPlan(5, (0,2); weighted=true)
        @test_throws ErrorException DiskVectorPlan(5, (2,1); weighted=true)
        @test_throws ErrorException DiskVectorIPlan(5, 0; weighted=true)
    end

    @testset "ComplexZernikeITransform: a=$a, N=$N" for a in (0, 1), N in (7, 8)
        Pⁱ = ComplexZernikeITransform(N, a)
        P = inv(Pⁱ)
        Ñ = N ÷ 2 + 1
        F = [2(k-1) + abs(_complexzernikemode(c)) ≤ N-1 ? cos(k + 2c) + im*sin(3k - c) : 0.0im for k = 1:Ñ, c = 1:4Ñ-3]
        vals = Pⁱ * F
        @test size(vals) == size(grid(Pⁱ))
        Z = ComplexZernike{ComplexF64}(0, a)
        for (k, j) in ((1,1), (2,3), (Ñ, 4Ñ-3))
            𝐱 = grid(Pⁱ)[k,j]
            @test vals[k,j] ≈ sum(Z[𝐱, Block.(1:N)] .* ModalTrav(F)[Block.(1:N)])
        end
        @test (P * vals).matrix ≈ F
        @test Pⁱ * F[1:2, 1:5] ≈ Pⁱ * [F[1:2, 1:5] zeros(2, 4Ñ-8); zeros(Ñ-2, 4Ñ-3)]
        F_high = zeros(ComplexF64, Ñ+1, 4Ñ+1)
        F_high[end, end] = F_high[Ñ, 4Ñ-3] = 1 # degrees above 2Ñ-2 are ignored
        @test Pⁱ * F_high == zero(vals)
        @test Pⁱ * (P * vals) ≈ vals # P fills the ignored entries with rounding errors
    end

    @testset "diskvector2disk inverts disk2diskvector" begin
        n, ncols = 5, 9
        F₁ = [cos(k + 2c) + im*sin(k*c) for k = 1:n, c = 1:ncols]
        F₂ = [sin(2k - c) + im*cos(k + c) for k = 1:n, c = 1:ncols]
        G = disk2diskvector(F₁, F₂)
        F̃₁, F̃₂ = diskvector2disk(G)
        @test size(F̃₁) == size(F̃₂) == (n, ncols+4)
        @test F̃₁ ≈ [F₁ zeros(n, 4)]
        @test F̃₂ ≈ [F₂ zeros(n, 4)]
        G = [cos(r + j) + im*sin(2r - j) for r = 1:2n, j = 1:ncols]
        @test disk2diskvector(diskvector2disk(G)...) ≈ [G zeros(2n, 4)]
    end

    @testset "DiskVectorIPlan: 𝐚=$𝐚, N=$N" for 𝐚 in weights, N in (7, 8)
        P = DiskVectorPlan(N, 𝐚)
        Pⁱ = inv(P)
        @test grid(Pⁱ) == grid(P)
        Ñ = N ÷ 2 + 1
        G = [_diskvectordegree(_complexzernikemode(j), r) ≤ N-1 ? cos(r + 2j) + im*sin(3r - j) : 0.0im for r = 1:2Ñ, j = 1:4Ñ-1]
        𝐅 = Pⁱ * G
        @test P * 𝐅 ≈ G
        for (k, j) in ((1,1), (2,3), (Ñ, 4Ñ-3))
            𝐱 = grid(P)[k,j]
            @test 𝐅[k,j] ≈ sum(sum(DiskVector(_complexzernikemode(c), 𝐚)[𝐱, 1:2Ñ] .* G[:,c]) for c in axes(G,2))
        end
        𝐟 = 𝐱 -> SVector(exp(𝐱[1])*cos(𝐱[2]), sin(𝐱[1] + 2𝐱[2]))
        P₄₁ = DiskVectorPlan(41, 𝐚)
        @test inv(P₄₁) * (P₄₁ * 𝐟.(grid(P₄₁))) ≈ 𝐟.(grid(P₄₁))
        @test Pⁱ * G[1:3, 1:5] ≈ Pⁱ * [G[1:3, 1:5] zeros(3, 4Ñ-6); zeros(2Ñ-3, 4Ñ-1)]
        G_high = zeros(ComplexF64, 2Ñ, 4Ñ-1)
        G_high[2, end] = 1 # type-2 element of degree 2Ñ in mode 2Ñ-1, which is ignored
        @test Pⁱ * (G + G_high) ≈ 𝐅
    end

    @testset "weighted DiskVectorPlan recovers every weighted field: 𝐚=$𝐚, N=$N" for 𝐚 in ((0,1), (1,0), (1,1)), N in (7, 8)
        P = DiskVectorPlan(N, 𝐚; weighted=true)
        𝐱 = grid(P)
        for m in -N:N, r in 1:2N
            _diskvectordegree(m, r, 𝐚) ≤ N-1 || continue
            W = Weighted(DiskVector(m, 𝐚))
            G = P * [W[x, r] for x in 𝐱]
            E = zeros(ComplexF64, size(G))
            E[r, _complexzernikecolumn(m)] = 1
            @test G ≈ E atol=1e-12
        end
    end

    @testset "weighted DiskVectorIPlan: 𝐚=$𝐚, N=$N" for 𝐚 in ((0,1), (1,0), (1,1)), N in (7, 8)
        P = DiskVectorPlan(N, 𝐚; weighted=true)
        Pⁱ = inv(P)
        @test Pⁱ.weighted
        @test grid(Pⁱ) == grid(P)
        Ñ = N ÷ 2 + 1
        G = [_diskvectordegree(_complexzernikemode(j), r, 𝐚) ≤ N-1 ? cos(r + 2j) + im*sin(3r - j) : 0.0im for r = 1:2Ñ, j = 1:4Ñ-1]
        𝐅 = Pⁱ * G
        @test P * 𝐅 ≈ G
        @test inv(Pⁱ) * 𝐅 ≈ G
        @test DiskVectorIPlan(N, 𝐚; weighted=true) * G ≈ 𝐅
        for (k, j) in ((1,1), (2,3), (Ñ, 4Ñ-3))
            𝐱 = grid(P)[k,j]
            @test 𝐅[k,j] ≈ sum(sum(Weighted(DiskVector(_complexzernikemode(c), 𝐚))[𝐱, r] * G[r,c] for r in axes(G,1)) for c in axes(G,2))
        end
        G_high = zeros(ComplexF64, 2Ñ, 4Ñ-1)
        G_high[𝐚[2] == 1 ? 2Ñ-2 : 2Ñ-3, 1] = 1 # a weighted field of mode 0, of degree 2Ñ-3, but 2Ñ-1 once weighted, which is ignored
        @test Pⁱ * (G + G_high) ≈ 𝐅
        # a field with no tangential component on the boundary, no normal component, or that vanishes there
        𝐟 = 𝐱 -> (1 - 𝐱[1]^2 - 𝐱[2]^2)^𝐚[1] * exp(𝐱[1])*cos(𝐱[2]) * SVector(𝐱[1], 𝐱[2]) + (1 - 𝐱[1]^2 - 𝐱[2]^2)^𝐚[2] * sin(𝐱[1]) * SVector(-𝐱[2], 𝐱[1]) +
            (1 - 𝐱[1]^2 - 𝐱[2]^2) * SVector(sin(𝐱[1] + 2𝐱[2]), 𝐱[1]^2)
        P₄₁ = DiskVectorPlan(41, 𝐚; weighted=true)
        @test inv(P₄₁) * (P₄₁ * 𝐟.(grid(P₄₁))) ≈ 𝐟.(grid(P₄₁))
    end

    @testset "mass matrices between DiskVector bases: m=$m" for m in (-2, 0, 1)
        V = DiskVector(m)
        Q = DiskVector(m, (0,-1))
        N = Weighted(DiskVector(m, (0,1)))
        @test (Q'V)[1:6,1:6] ≈ [sum(Q[𝐱,k]'V[𝐱,j] for 𝐱 in UnitDisk()) for k=1:6, j=1:6]
        @test (V'Q)[1:6,1:6] ≈ (Q'V)[1:6,1:6]'
        @test (N'V)[1:6,1:6] ≈ [sum(N[𝐱,k]'V[𝐱,j] for 𝐱 in UnitDisk()) for k=1:6, j=1:6]
        @test (V'N)[1:6,1:6] ≈ (N'V)[1:6,1:6]'
        @test (N'Q)[1:6,1:6] ≈ [sum(N[𝐱,k]'Q[𝐱,j] for 𝐱 in UnitDisk()) for k=1:6, j=1:6]
        @test (Q'Q)[1:6,1:6] ≈ [sum(Q[𝐱,k]'Q[𝐱,j] for 𝐱 in UnitDisk()) for k=1:6, j=1:6]
        B = Weighted(DiskVector(m, (1,1)))
        @test (B'V)[1:6,1:6] ≈ [sum(B[𝐱,k]'V[𝐱,j] for 𝐱 in UnitDisk()) for k=1:6, j=1:6]
        @test (B'B)[1:6,1:6] ≈ [sum(B[𝐱,k]'B[𝐱,j] for 𝐱 in UnitDisk()) for k=1:6, j=1:6]
    end
end
