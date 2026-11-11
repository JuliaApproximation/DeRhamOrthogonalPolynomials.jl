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

    @testset "boundary conditions" begin
        # Weighted(DiskVector(m, (0,1))) has no tangential component on the boundary,
        # Weighted(DiskVector(m, (1,0))) has no normal component on the boundary,
        # Weighted(DiskVector(m, (1,1))) vanishes on the boundary.
        # Note that column j of the coefficients holds the mode m = (-1)^iseven(j) * j ÷ 2

        h = 𝐱 -> exp(𝐱[1]) * cos(𝐱[2])
        𝐠 = 𝐱 -> SVector(cos(4𝐱[1] + 𝐱[2]), exp(𝐱[1]) * sin(𝐱[1] - 3𝐱[2]))
        b = 𝐱 -> 1 - 𝐱[1]^2 - 𝐱[2]^2 # vanishes on the boundary
        normal = (𝐯, 𝐲) -> dot(𝐲, 𝐯)
        tangential = (𝐯, 𝐲) -> 𝐲[1]*𝐯[2] - 𝐲[2]*𝐯[1]

        𝐲 = SVector(cos(0.7), sin(0.7))

        weights = ((0, 1), (1, 0), (1, 1))
        fields = (𝐱 -> h(𝐱)*𝐱 + b(𝐱)*𝐠(𝐱),                   # normal on the boundary
            𝐱 -> h(𝐱)*SVector(-𝐱[2], 𝐱[1]) + b(𝐱)*𝐠(𝐱), # tangential on the boundary
            𝐱 -> b(𝐱)*𝐠(𝐱))                             # zero on the boundary
        conditions = ((tangential,), (normal,), (normal, tangential)) # components that vanish on the boundary

        for (𝐚, 𝐟, condition) in zip(weights, fields, conditions)
            # the transform is exact up to rounding for a field of high enough degree
            Q = DiskVectorPlan(41, 𝐚; weighted=true)
            @test inv(Q) * (Q * 𝐟.(grid(Q))) ≈ 𝐟.(grid(Q))

            # with only a few coefficients the boundary condition still holds, unlike for the unweighted basis
            P, P₀ = DiskVectorPlan(9, 𝐚; weighted=true), DiskVectorPlan(9)
            G, G₀ = P * 𝐟.(grid(P)), P₀ * 𝐟.(grid(P₀))
            𝐟ₙ = sum(Weighted(DiskVector((-1)^iseven(j) * j ÷ 2, 𝐚))[𝐲, r] * G[r, j] for r in axes(G, 1), j in axes(G, 2))
            𝐟₀ = sum(DiskVector((-1)^iseven(j) * j ÷ 2)[𝐲, r] * G₀[r, j] for r in axes(G₀, 1), j in axes(G₀, 2))
            for c in condition
                @test abs(c(𝐟ₙ, 𝐲)) ≤ 1e-14
                @test abs(c(𝐟₀, 𝐲)) > 1e-5
            end
        end
    end

    @testset "boundary values" begin
        # The boundary values of a vector field can be read off its coefficients in the bases with a = -1.
        M = 40 # total degree
        # column j of the coefficients holds the mode m = (-1)^iseven(j) * j ÷ 2

        𝐟 = 𝐱 -> SVector(cos(4𝐱[1] + 𝐱[2]), exp(𝐱[1]) * sin(𝐱[1] - 3𝐱[2]))
        f_r = θ -> (𝐲=SVector(cos(θ), sin(θ)); dot(𝐲, 𝐟(𝐲)))               # normal component on the boundary
        f_θ = θ -> (𝐲=SVector(cos(θ), sin(θ)); 𝐲[1]*𝐟(𝐲)[2] - 𝐲[2]*𝐟(𝐲)[1]) # tangential component on the boundary
        θ = range(0, 2π; length=97)

        # Only the first two elements of DiskVector(m, (-1,-1)) are nonzero on the boundary. For m ≠ 0 they are exp(im*m*θ)
        # times (1, im*sign(m)) and (1, -im*sign(m)) in polar components, and for m = 0 they are 𝐞_r and 𝐞_θ.
        P = DiskVectorPlan(M+1, -1)
        G = P * 𝐟.(grid(P))
        u_r = θ -> sum((m=(-1)^iseven(j) * j ÷ 2; (iszero(m) ? G[1, j] : G[1, j] + G[2, j]) * cis(m*θ)) for j in axes(G, 2))
        u_θ = θ -> sum((m=(-1)^iseven(j) * j ÷ 2; (iszero(m) ? G[2, j] : im*sign(m)*(G[1, j] - G[2, j])) * cis(m*θ)) for j in axes(G, 2))
        @test u_r.(θ) ≈ f_r.(θ)
        @test u_θ.(θ) ≈ f_θ.(θ)

        # so a field that vanishes on the boundary has no coefficients in the first two rows
        @test norm((P*[(1-𝐱[1]^2-𝐱[2]^2)*𝐟(𝐱) for 𝐱 in grid(P)])[1:2, :]) ≤ 1e-13

        # Only the first element of DiskVector(m, (-1,0)) has a normal component on the boundary, and only the first
        # (the second for m = 0) of DiskVector(m, (0,-1)) has a tangential one.
        G_n = DiskVectorPlan(M+1, (-1, 0)) * 𝐟.(grid(P))
        G_t = DiskVectorPlan(M+1, (0, -1)) * 𝐟.(grid(P))
        # the normal and tangential components on the boundary, each from a single row
        n_r = θ -> sum((m=(-1)^iseven(j) * j ÷ 2; (iszero(m) ? G_n[1, j] : G_n[1, j]/2) * cis(m*θ)) for j in axes(G_n, 2))
        t_θ = θ -> sum((m=(-1)^iseven(j) * j ÷ 2; (iszero(m) ? G_t[2, j] : im*sign(m)*G_t[1, j]/2) * cis(m*θ)) for j in axes(G_t, 2))
        @test n_r.(θ) ≈ f_r.(θ)
        @test t_θ.(θ) ≈ f_θ.(θ)
    end

    @testset "derivatives" begin
        # grad and curl act mode by mode, so the coefficients of the derivatives follow from those of the field:
        #
        #   grad: DiskBasis(m) → DiskVector(m, (1,0)),   curl: DiskVector(m, (1,0)) → DiskBasis(m, 1)

        N = 41 # total degree N-1
        # column j of the coefficients holds the mode m = (-1)^iseven(j) * j ÷ 2

        u = 𝐱 -> exp(𝐱[1]) * sin(2𝐱[2]) + cos(3𝐱[1]*𝐱[2])
        ∇u = 𝐱 -> SVector(exp(𝐱[1]) * sin(2𝐱[2]) - 3𝐱[2] * sin(3𝐱[1]*𝐱[2]), 2exp(𝐱[1]) * cos(2𝐱[2]) - 3𝐱[1] * sin(3𝐱[1]*𝐱[2]))
        𝐟 = 𝐱 -> SVector(cos(4𝐱[1] + 𝐱[2]), exp(𝐱[1]) * sin(𝐱[1] - 3𝐱[2]))
        curl𝐟 = 𝐱 -> exp(𝐱[1]) * (sin(𝐱[1] - 3𝐱[2]) + cos(𝐱[1] - 3𝐱[2])) + sin(4𝐱[1] + 𝐱[2])

        # grad: the scalar transform gives the coefficients of u in DiskBasis(m)
        S = plan_transform(ComplexZernike{ComplexF64}(), (Block(N),))
        C = (S * u.(grid(ComplexZernike{ComplexF64}(), Block(N)))).matrix
        n = size(C,1)
        G = zeros(ComplexF64, 2n+2, size(C,2))
        for j in axes(C,2) # Put coefficients of gradu into G
            G[:,j] = coefficients(grad(DiskBasis((-1)^iseven(j) * j ÷ 2)))[1:2n+2, 1:n] * C[:,j]
        end
        Pⁱ = DiskVectorIPlan(N+2, (1,0)) # evaluates the coefficients on a grid
        @test Pⁱ * G ≈ ∇u.(grid(Pⁱ))

        # curl: the vector transform gives the coefficients of 𝐟 in DiskVector(m, (1,0))
        P = DiskVectorPlan(N, (1,0))
        F = P * 𝐟.(grid(P))
        H = zeros(ComplexF64, size(F,1) ÷ 2 + 1, size(F,2))
        for j in axes(F,2) # Put coefficients of curl𝐟 into H
            H[:,j] = coefficients(curl(DiskVector((-1)^iseven(j) * j ÷ 2, (1,0))))[axes(H,1), axes(F,1)] * F[:,j]
        end
        Sⁱ = ComplexZernikeITransform(N+2, 1) # evaluates the coefficients in DiskBasis(m, 1) on a grid
        @test Sⁱ * H ≈ curl𝐟.(grid(Sⁱ))

        # and curl(grad(u)) == 0 holds exactly for the coefficient matrices
        for m in -3:3
            D = coefficients(grad(DiskBasis(m)))[1:2n+2, 1:n]
            K = coefficients(curl(DiskVector(m, (1,0))))[1:n+1, 1:2n+2]
            @test norm(K * D) ≤ 1e-12 * norm(D)
        end
    end
end
