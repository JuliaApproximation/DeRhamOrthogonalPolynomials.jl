using DeRhamOrthogonalPolynomials, MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, ContinuumArrays, StaticArrays, BlockArrays, LinearAlgebra, Test
using DeRhamOrthogonalPolynomials: _diskmatrixdegree, _complexzernikecolumn, equivariantjacobimatrix

@testset "DiskMatrix transform" begin
    ρ = θ -> @SMatrix [cos(θ) -sin(θ); sin(θ) cos(θ)] # rotation by θ
    Σ = 𝐱 -> @SMatrix [𝐱[1]^2-𝐱[2]^2 2𝐱[1]*𝐱[2]; 2𝐱[1]*𝐱[2] 𝐱[2]^2-𝐱[1]^2]
    α,δ,β,γ = t -> exp(t) + 2, t -> cos(3t), t -> sin(2t) + 1, t -> 1/(2+t)
    A = 𝐱 -> (t = norm(𝐱)^2; α(t)*I + δ(t)*ρ(π/2) + β(t)*Σ(𝐱) + γ(t)*ρ(π/2)*Σ(𝐱))
    C = (𝐱, r) -> begin # the term with coefficient g[r], as in disk2diskmatrix
        k, ν, t = (r-1) ÷ 4, (r-1) % 4, 𝐱[1]^2 + 𝐱[2]^2
        ν < 2 ? jacobip(k, 0, 0, 2t-1) * ρ(ν*π/2) : jacobip(k, 0, 2, 2t-1) * ρ((ν-2)*π/2) * Σ(𝐱)
    end

    @testset "layout" begin
        @test [_diskmatrixdegree(r) for r in 1:10] == [0,0,2,2,2,2,4,4,4,4]

        @testset "the complex matrix Zernike polynomials of mode 0 are a recombination" begin
            𝐱 = SVector(0.3, -0.45)
            𝐲₊,𝐲₋ = SVector(1,im),SVector(1,-im)
            for j in 2:5
                z₋,z₀,z₊ = DiskBasis(-2)[𝐱,j-1],DiskBasis(0)[𝐱,j],DiskBasis(2)[𝐱,j-1]
                k = j-1
                @test z₋*𝐲₊*transpose(𝐲₊) ≈ C(𝐱,4k-1) + im*C(𝐱,4k)
                @test z₊*𝐲₋*transpose(𝐲₋) ≈ C(𝐱,4k-1) - im*C(𝐱,4k)
                @test z₀*𝐲₋*transpose(𝐲₊) ≈ C(𝐱,4j-3) - im*C(𝐱,4j-2)
                @test z₀*𝐲₊*transpose(𝐲₋) ≈ C(𝐱,4j-3) + im*C(𝐱,4j-2)
            end
        end
    end

    @testset "DiskMatrixPlan recovers every element: N=$N" for N in (5, 6, 7, 8)
        P = DiskMatrixPlan(N)
        𝐱 = grid(P)
        Ñ = N ÷ 2 + 1
        for r in 1:4Ñ
            _diskmatrixdegree(r) ≤ N-1 || continue
            g = P * [C(x,r) for x in 𝐱]
            @test length(g) == 4Ñ
            E = zeros(ComplexF64, 4Ñ)
            E[r] = 1
            @test g ≈ E atol=1e-12
        end
    end

    @testset "disk2diskmatrix from ComplexZernike coefficients" begin
        M = 40
        f₁₁ = [A(SVector(x,y))[1,1] for (x,y) in UnitDisk()]
        f₁₂ = [A(SVector(x,y))[1,2] for (x,y) in UnitDisk()]
        f₂₁ = [A(SVector(x,y))[2,1] for (x,y) in UnitDisk()]
        f₂₂ = [A(SVector(x,y))[2,2] for (x,y) in UnitDisk()]
        F₁₁ = ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ f₁₁
        F₁₂ = ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ f₁₂
        F₂₁ = ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ f₂₁
        F₂₂ = ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ f₂₂
        g = disk2diskmatrix(F₁₁, F₁₂, F₂₁, F₂₂)
        @test length(g) == 4size(F₁₁.matrix,1)
        @test g == disk2diskmatrix(F₁₁.matrix, F₁₂.matrix, F₂₁.matrix, F₂₂.matrix)
        for 𝐱 in (SVector(0.1,0.2), SVector(-0.5,0.3), SVector(0.0,0.0), SVector(0.3,-0.6))
            @test sum(C(𝐱,r)*g[r] for r in eachindex(g)) ≈ A(𝐱)
        end
        P = DiskMatrixPlan(M+1)
        @test P * [A(SVector(𝐱[1],𝐱[2])) for 𝐱 in grid(P)] ≈ g
        @test_throws AssertionError disk2diskmatrix(F₁₁.matrix, F₁₂.matrix, F₂₁.matrix, F₂₂.matrix[:,1:end-1])
    end

    @testset "radial coefficients" begin
        P = DiskMatrixPlan(41)
        g = P * [A(SVector(𝐱[1],𝐱[2])) for 𝐱 in grid(P)]
        n = length(g) ÷ 4
        for t in (0.0, 0.3, 0.8, 1.0)
            @test sum(g[4k-3]*jacobip(k-1,0,0,2t-1) for k in 1:n) ≈ α(t)
            @test sum(g[4k-2]*jacobip(k-1,0,0,2t-1) for k in 1:n) ≈ δ(t)
            @test sum(g[4k-1]*jacobip(k-1,0,2,2t-1) for k in 1:n) ≈ β(t)
            @test sum(g[4k]*jacobip(k-1,0,2,2t-1) for k in 1:n) ≈ γ(t)
        end
        @test norm(imag(g)) ≤ 1e-13

        @testset "real symmetric has δ = 0" begin
            W = 𝐱 -> (t = norm(𝐱)^2; α(t)*I + β(t)*Σ(𝐱) + γ(t)*ρ(π/2)*Σ(𝐱))
            @test all(𝐱 -> W(𝐱) ≈ W(𝐱)', (SVector(0.3,0.2), SVector(-0.1,0.7)))
            @test norm((P * [W(SVector(𝐱[1],𝐱[2])) for 𝐱 in grid(P)])[2:4:end]) ≤ 1e-13
        end
    end

    @testset "parts that are not equivariant are discarded" begin
        N = 9
        P = DiskMatrixPlan(N)
        B = 𝐱 -> sum(cos(r)*C(𝐱,r) for r in 1:18)
        Bs = [B(SVector(𝐱[1],𝐱[2])) for 𝐱 in grid(P)]
        S = @SMatrix [1 0; 0 -1]
        H = @SMatrix [0 1; 1 0]
        for Y in (𝐱 -> S, 𝐱 -> H, 𝐱 -> 𝐱[1]*ρ(π/2), 𝐱 -> (@SMatrix [𝐱[1] 𝐱[2]; 𝐱[2] -𝐱[1]]), 𝐱 -> 𝐱[1]*𝐱[2]*I, 𝐱 -> real((𝐱[1]+im*𝐱[2])^4)*Σ(𝐱))
            @test P * (Bs + [Y(SVector(𝐱[1],𝐱[2])) for 𝐱 in grid(P)]) ≈ P * Bs
        end

        @testset "giving the Frobenius projection" begin
            f = 𝐱 -> @SMatrix [exp(𝐱[1]) sin(𝐱[2]); 𝐱[1]*𝐱[2] cos(𝐱[1]+𝐱[2])]
            Q = DiskMatrixPlan(41)
            g = Q * [f(SVector(𝐱[1],𝐱[2])) for 𝐱 in grid(Q)]
            for r in 1:8
                @test g[r] ≈ sum(tr(C(𝐱,r)'f(SVector(𝐱[1],𝐱[2]))) for 𝐱 in UnitDisk())/sum(tr(C(𝐱,r)'C(𝐱,r)) for 𝐱 in UnitDisk()) atol=1e-12
            end
        end
    end

    @testset "diskmatrix2disk inverts disk2diskmatrix" begin
        n, ncols = 5, 9
        g = [cos(r) + im*sin(2r) for r = 1:4n]
        F₁₁,F₁₂,F₂₁,F₂₂ = diskmatrix2disk(g)
        @test size(F₁₁) == size(F₁₂) == size(F₂₁) == size(F₂₂) == (n, _complexzernikecolumn(2))
        @test disk2diskmatrix(F₁₁, F₁₂, F₂₁, F₂₂) ≈ g
        @test diskmatrix2disk(g[1:4n-1])[1][n,1] == g[4n-3]
        @test diskmatrix2disk(g[1:4n-1])[1][n,4] == g[4n-1]/2
    end

    @testset "DiskMatrixIPlan: N=$N" for N in (7, 8)
        P = DiskMatrixPlan(N)
        Pⁱ = inv(P)
        @test Pⁱ isa DiskMatrixIPlan
        @test inv(Pⁱ) isa DiskMatrixPlan
        @test grid(Pⁱ) == grid(P)
        Ñ = N ÷ 2 + 1
        g = [_diskmatrixdegree(r) ≤ N-1 ? cos(r) + im*sin(3r) : 0.0im for r = 1:4Ñ]
        𝐅 = Pⁱ * g
        @test size(𝐅) == size(grid(P))
        @test P * 𝐅 ≈ g
        @test DiskMatrixIPlan(N) * g == 𝐅
        for (k, j) in ((1,1), (2,3), (Ñ, 4Ñ-3))
            𝐱 = grid(P)[k,j]
            @test 𝐅[k,j] ≈ sum(C(𝐱,r)*g[r] for r in 1:4Ñ)
        end
        @test Pⁱ * g[1:6] ≈ Pⁱ * [g[1:6]; zeros(4Ñ-6)]
        g_high = zeros(ComplexF64, 4Ñ)
        g_high[end] = 1 # ρ(π/2)*Σ with degree 2Ñ, which is ignored
        @test Pⁱ * (g + g_high) ≈ 𝐅
    end

    @testset "round trip" begin
        P = DiskMatrixPlan(41)
        𝐅 = [A(SVector(𝐱[1],𝐱[2])) for 𝐱 in grid(P)]
        @test inv(P) * (P * 𝐅) ≈ 𝐅
    end

    @testset "equivariantmultiplication: m=$m" for m in -3:3
        Q = DiskVector(m)
        𝐱 = SVector(0.3, -0.45)
        P = DiskMatrixPlan(41)
        g = P * [A(SVector(𝐱[1],𝐱[2])) for 𝐱 in grid(P)]
        M = equivariantmultiplication(Q, g)
        N = 70
        Mₙ = M[1:N,1:N]
        for j in 1:14
            @test A(𝐱)*Q[𝐱,j] ≈ sum(Q[𝐱,i]*Mₙ[i,j] for i in 1:N)
        end

        n = 12
        @test equivariantmultiplication(Q, [1.0])[1:n,1:n] ≈ I
        @test equivariantmultiplication(Q, [0.5,0,0,0,0.5])[1:n,1:n] ≈ equivariantjacobimatrix(Val(1), Q)[1:n,1:n] # t = (P₀ + P₁)/2
        @test equivariantmultiplication(Q, [0,0,1.0])[1:n,1:n] ≈ equivariantjacobimatrix(Val(2), Q)[1:n,1:n]
        Mⱼ = equivariantmultiplication(Q, [0,1.0])[1:n,1:n]
        @test Mⱼ'Mⱼ ≈ I
        for j in 1:n
            @test ρ(π/2)*Q[𝐱,j] ≈ sum(Q[𝐱,i]*Mⱼ[i,j] for i in 1:n)
        end
        @test iszero(equivariantmultiplication(Q, Float64[])[1:n,1:n])
    end

    @test_throws AssertionError equivariantmultiplication(DiskVector(1, (0,1)), [1.0])
end
