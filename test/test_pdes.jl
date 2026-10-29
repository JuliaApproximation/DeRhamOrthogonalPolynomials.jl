using DeRhamOrthogonalPolynomials, ClassicalOrthogonalPolynomials, BlockArrays, SpecialFunctions, FastGaussQuadrature, StaticArrays, BlockBandedMatrices, BandedMatrices, SparseArrays, Test

@testset "PDEs" begin
    @testset "Disk" begin
        @testset "Laplace eigenvalues" begin
            @testset "Dirichlet" begin
                @testset "eigenvalue" begin
                    for m = -3:3
                        P = DiskBasis(m)
                        W = Weighted(DiskBasis(m,1))
                        Δ = grad(W)'grad(W)
                        M = W'W
                        @test (P\W)[1:10,1:10] ≈ [sum(P[𝐱,k]'W[𝐱,j] for 𝐱 in UnitDisk())/sum(P[𝐱,k]'P[𝐱,k] for 𝐱 in UnitDisk()) for k=1:10, j=1:10]
                        @test M[1:10,1:10] ≈ [sum(W[𝐱,k]'W[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10, j=1:10]
                        @test Δ[1:10,1:10] ≈ [sum(grad(W)[:,k]'grad(W)[:,j]) for k=1:10,j=1:10]

                        n = 20
                        λ,U = eigen(Symmetric(Diagonal(view(Δ,1:n,1:n))), Symmetric(M[1:n,1:n]))
                        @test λ[1:5] ≈ approx_besselroots(abs(m),5).^2
                        𝐱 = SVector(0.1,0.2); θ = atan(reverse(𝐱)...); r = norm(𝐱)
                        u = (W[:,1:n]U[:,1])
                        @test u[𝐱] * besselj(abs(m), sqrt(λ[1])/2)/u[SVector(0.5,0)] ≈ exp(im*m*θ)*besselj(abs(m), sqrt(λ[1])*r)
                    end
                end

                @testset "Poisson" begin
                    
                end
            end

            @testset "Neumann" begin
                for m = -3:3
                    P = DiskBasis(m)
                    Q = DiskBasis(m, -1)
                    Δ = grad(Q)'grad(Q)
                    M = Q'Q
                    @test (P\Q)[1:10,1:10] ≈ [sum(P[𝐱,k]'Q[𝐱,j] for 𝐱 in UnitDisk())/sum(P[𝐱,k]'P[𝐱,k] for 𝐱 in UnitDisk()) for k=1:10, j=1:10]
                    @test M[1:10,1:10] ≈ [sum(Q[𝐱,k]'Q[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10, j=1:10]
                    @test Δ[1:10,1:10] ≈ [sum(grad(Q)[:,k]'grad(Q)[:,j]) for k=1:10,j=1:10]

                    n = 20
                    λ,U = eigen(Symmetric(Diagonal(view(Δ,1:n,1:n))), Symmetric(M[1:n,1:n]))
                    λ_approx = findall(iszero, diff([besselj(abs(m), x) for x in sqrt(max(0,λ[1]-0.1))..sqrt(λ[5]+0.1)])) .^ 2
                    @test λ[1:5] ≈ λ_approx
                    𝐱 = SVector(0.1,0.2); θ = atan(reverse(𝐱)...); r = norm(𝐱)
                    u = (Q[:,1:n]U[:,1])
                    @test u[𝐱] * besselj(abs(m), sqrt(λ[1]))/u[SVector(1,0)] ≈ exp(im*m*θ)*besselj(abs(m), sqrt(λ[1])*r)
                end
            end
        end

        @testset "curl" begin
            for m = -3:3
                N = Weighted(DiskVector(m, (0,1)))
                @test (N'N)[1:10,1:10] ≈ [sum(N[𝐱,k]'N[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10,j=1:10]
                @test (curl(N)'curl(N))[1:10,1:10] ≈ [sum(curl(N)[𝐱,k]'curl(N)[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10, j=1:10]

                Q = DiskVector(m, (0,-1))
                @test (Q'Q)[1:10,1:10] ≈ [sum(Q[𝐱,k]'Q[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10,j=1:10]
                @test (curl(Q)'curl(Q))[1:10,1:10] ≈ [sum(curl(Q)[𝐱,k]'curl(Q)[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10, j=1:10]
            end
        end

        @testset "k = 2" begin
            @testset "Natural (Dirchlet)" begin
                n = 30
                for m = -4:4
                    Q = DiskVector(m, (0, -1))
                    P = DiskBasis(m)
                    JR = Block.(1:n)
                    KR = Block.(1:(n-!iszero(m)))
                    D_Q = sparse((P'curl(Q))[KR,JR])
                    M_Q = sparse((Q'Q)[JR,JR])
                    A = [-M_Q D_Q'; D_Q 0I]
                    M_P = sparse((P'P)[KR,KR])
                    B = [-zero(M_Q) zero(D_Q)'; zero(D_Q) M_P]
                    λ = sort!(abs.(filter!(!isnan, eigvals(Matrix(A), Matrix(B)))))
                    @test λ[1:5] ≈ approx_besselroots(abs(m),5).^2
                end
            end
            @testset "Neumann" begin
                n = 30
                for m = -4:4
                    P = DiskBasis(m)
                    N = Weighted(DiskVector(m, (0,1)))
                    JR = Block.(1:n)
                    KR = Block.(1:(n-!iszero(m)))

                    M_N = (N'N)[JR,JR]
                    D_N = Matrix((P'curl(N))[1:n,JR])
                    M_P = (P'P)[1:n,1:n]
                    Z = zero(M_P)
                    A_N = [-M_N D_N'; D_N Z]
                    Z₂ = zero(M_N)
                    Z₃ = zero(D_N)
                    B = [Z₂ Z₃'; Z₃ M_P]
                    λ_N = eigvals(A_N, B)
                    @test filter(x -> real(x)>0, λ_N)[1:3] ≈ findall(iszero, diff([besselj(abs(m),r) for r in 0.01..13]))[1:3].^2
                end
            end
        end
    end

    @testset "Cylinder" begin
        @testset "Weak form" begin
            @testset "Dirichlet" begin
                for m = -2:2, k = -2:2
                    W = Weighted(DiskBasis(m, 1))
                    M = 2π * (W'W)
                    Δ = 2π * grad(W)'grad(W) + k^2 * M

                    # We can compute the low order eigenvalues/functions to high accuracy:
                    n = 20
                    λ,Q = eigen(Symmetric(Δ[1:n,1:n]), Symmetric(M[1:n,1:n]))

                    # the eigenvalues are shifted:
                    @test λ[1:4] ≈ findall(iszero, [besselj(m,x) for x in 1..20])[1:4] .^ 2 .+ k^2
                end
            end

            @testset "Neumann" begin
                for m = -2:2, k = -2:2
                    W = DiskBasis(m, -1)
                    M = 2π * (W'W)
                    Δ = 2π * grad(W)'grad(W) + k^2 * M

                    # We can compute the low order eigenvalues/functions to high accuracy:
                    n = 20
                    λ,Q = eigen(Symmetric(Δ[1:n,1:n]), Symmetric(M[1:n,1:n]))

                    # the eigenvalues are shifted:
                    @test λ[iszero(m) .+ (1:4)] ≈ findall(iszero, diff([besselj(m,x) for x in 1..20]))[1:4] .^ 2 .+ k^2
                    if iszero(m)
                        # The function exp(im*k*z) has gradient [0,0,exp(im*k*z)] which is tangential on the boundary
                        # so is an eigenfunction with eigenvalue k^2. 
                        @test all(iszero,Q[2:end,1])
                        @test λ[1] == abs(k)^2
                    end
                end
            end
        end

        @testset "Maxwell Neumann" begin
            m,ω = 1,2
            Q = DiskBasis(m, -1)
            V = DiskVector(m, (0,-1))

            n = 20
            KR = Block.(1:n)        # 𝐧 DOF
            JR = Block.(1:n-1)      # w DOF
            D = (V'grad(Q))[KR,JR]
            M_N = (V'V)[KR,KR]
            M_W = (Q'Q)[JR,JR]
            Δ_W = (grad(Q)'grad(Q))[JR,JR]
            C_N = (curl(V)'curl(V))[collect(axes(M_N,1)), collect(axes(M_N,1))]
            M = 2π * [M_N                     zero(D)          ;
                                zero(D')                M_W]

            @test M' ≈ M

            # [sum(N[𝐱,k]'N[𝐱,j] for 𝐱 in UnitDisk()) for k in axes(M_N,1),  j in axes(M_N,1)]


            C = 2π * [ω^2*M_N+C_N    -im*ω*D          ;
                                im*ω*D'             Δ_W       ]
            @test C ≈ C'
            @test M ≈ M'

            # using CairoMakie; CairoMakie.activate!()
            # fig = Figure()
            # ax = Axis(fig[1, 1]; yreversed = true)
            # spy!(ax, abs.(C))
            # fig

            # fig = Figure()
            # ax = Axis(fig[1, 1]; yreversed = true)
            # spy!(ax, abs.(M))
            # fig

            # spy(abs.(M))

            λ,V = eigen(Hermitian(C), Hermitian(M))
        end
    end
end
