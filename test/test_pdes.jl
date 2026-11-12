using DeRhamOrthogonalPolynomials, ClassicalOrthogonalPolynomials, MultivariateOrthogonalPolynomials, BlockArrays, SpecialFunctions, FastGaussQuadrature, StaticArrays, BlockBandedMatrices, BandedMatrices, SparseArrays, LinearAlgebra, ForwardDiff, Test
using ContinuumArrays: grid

@testset "PDEs" begin
    @testset "Disk" begin
        @testset "load matrix: m=$m" for m in -3:3
            W = Weighted(DiskBasis(m,1))
            P = DiskBasis(m)
            @test (W'P)[1:6,1:6] ≈ [sum(W[𝐱,k]'P[𝐱,j] for 𝐱 in UnitDisk()) for k=1:6, j=1:6]
        end

        @testset "Sturm–Liouville" begin
            ρ = θ -> @SMatrix [cos(θ) -sin(θ); sin(θ) cos(θ)] # rotation by θ
            Σ = 𝐱 -> @SMatrix [𝐱[1]^2-𝐱[2]^2 2𝐱[1]*𝐱[2]; 2𝐱[1]*𝐱[2] 𝐱[2]^2-𝐱[1]^2]
            A = 𝐱 -> (t = sum(abs2,𝐱); (2+exp(t))*I + cos(3t)/2*ρ(π/2) + sin(2t)/2*Σ(𝐱) + ρ(π/2)*Σ(𝐱)/(2+t))
            P = DiskMatrixPlan(41)
            g = P * [A(SVector(𝐱[1],𝐱[2])) for 𝐱 in grid(P)]

            @testset "stiffness matrix: m=$m" for m in -2:2
                W = Weighted(DiskBasis(m,1))
                V = DiskVector(m)
                D = V \ grad(W)
                G = grammatrix(V)
                @test (D'*G*equivariantmultiplication(V, g)*D)[1:5,1:5] ≈ [sum(grad(W)[𝐱,k]'*(A(𝐱)*grad(W)[𝐱,j]) for 𝐱 in UnitDisk()) for k=1:5, j=1:5]
                @test (D'*G*equivariantmultiplication(V, [1.0])*D)[1:5,1:5] ≈ (grad(W)'grad(W))[1:5,1:5]
            end

            @testset "manufactured solution" begin
                u = 𝐱 -> (1-sum(abs2,𝐱))*exp(𝐱[1])*cos(2𝐱[2]+1)
                f = 𝐱 -> -tr(ForwardDiff.jacobian(𝐲 -> A(𝐲)*ForwardDiff.gradient(u,𝐲), 𝐱))
                M = 30
                F = (ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ [f(SVector(x,y)) for (x,y) in UnitDisk()]).matrix
                n = size(F,1)
                U = zeros(ComplexF64, n, 2M+1)
                for j = 1:2M+1
                    m = (-1)^iseven(j) * j ÷ 2
                    W = Weighted(DiskBasis(m,1))
                    V = DiskVector(m)
                    D = V \ grad(W)
                    S = (D'*grammatrix(V)*equivariantmultiplication(V, g)*D)[1:n,1:n]
                    U[:,j] = S \ ((W'DiskBasis(m))[1:n,1:n] * F[:,j])
                end
                for 𝐱 in (SVector(0.1,0.2), SVector(-0.5,0.3), SVector(0.0,0.0), SVector(0.3,-0.6), SVector(0.7,0.7))
                    @test sum((m = (-1)^iseven(j) * j ÷ 2; transpose(Weighted(DiskBasis(m,1))[𝐱,1:n])U[:,j]) for j = 1:2M+1) ≈ u(𝐱) atol=1e-13
                end
            end

            @testset "Neumann" begin
                @testset "stiffness matrix: m=$m" for m in -2:2
                    Q = DiskBasis(m, -1)
                    V = DiskVector(m)
                    D = V \ grad(Q)
                    G = grammatrix(V)
                    @test (D'*G*equivariantmultiplication(V, g)*D)[1:5,1:5] ≈ [sum(grad(Q)[𝐱,k]'*(A(𝐱)*grad(Q)[𝐱,j]) for 𝐱 in UnitDisk()) for k=1:5, j=1:5]
                    @test (D'*G*equivariantmultiplication(V, [1.0])*D)[1:5,1:5] ≈ (grad(Q)'grad(Q))[1:5,1:5]
                end

                @testset "manufactured solution" begin
                    # on the boundary 𝐱'A∇u = (α+β)*∂u/∂r + (γ-δ)*∂u/∂θ, which vanishes for each term of u
                    a, c = 2 + exp(1) + sin(2)/2, 1/3 - cos(3)/2
                    κ = (1 + im*c/a)/2
                    u = 𝐱 -> (t = sum(abs2,𝐱); cos(π*t) + (1-t)^2*exp(𝐱[1])*cos(2𝐱[2]+1) + (𝐱[1]+im*𝐱[2])*(1 + κ*(1-t)))
                    f = 𝐱 -> -tr(ForwardDiff.jacobian(𝐲 -> A(𝐲)*ForwardDiff.gradient(𝐳 -> real(u(𝐳)), 𝐲), 𝐱)) -
                            im*tr(ForwardDiff.jacobian(𝐲 -> A(𝐲)*ForwardDiff.gradient(𝐳 -> imag(u(𝐳)), 𝐲), 𝐱))
                    M = 36
                    F = (ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ [f(SVector(x,y)) for (x,y) in UnitDisk()]).matrix
                    n = size(F,1)
                    U = zeros(ComplexF64, n, 2M+1)
                    for j = 1:2M+1
                        m = (-1)^iseven(j) * j ÷ 2
                        Q = DiskBasis(m, -1)
                        V = DiskVector(m)
                        D = V \ grad(Q)
                        kr = (1+iszero(m)):n # the constant is in the kernel
                        S = (D'*grammatrix(V)*equivariantmultiplication(V, g)*D)[kr,kr]
                        U[kr,j] = S \ ((Q'DiskBasis(m))[kr,1:n] * F[:,j])
                    end
                    uₙ = 𝐱 -> sum((m = (-1)^iseven(j) * j ÷ 2; transpose(DiskBasis(m,-1)[𝐱,1:n])U[:,j]) for j = 1:2M+1)
                    𝐨 = SVector(0.0,0.0)
                    for 𝐱 in (SVector(0.1,0.2), SVector(-0.5,0.3), SVector(0.3,-0.6), SVector(0.7,0.7), SVector(1.0,0.0))
                        @test uₙ(𝐱) - uₙ(𝐨) ≈ u(𝐱) - u(𝐨) atol=1e-13
                    end
                end
            end
        end

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

        @testset "curl m = $m" for m = -3:3
            N = Weighted(DiskVector(m, (0,1)))
            @test (N'N)[1:10,1:10] ≈ [sum(N[𝐱,k]'N[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10,j=1:10]
            @test (curl(N)'curl(N))[1:10,1:10] ≈ [sum(curl(N)[𝐱,k]'curl(N)[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10, j=1:10]

            Q = DiskVector(m, (0,-1))
            @test (Q'Q)[1:10,1:10] ≈ [sum(Q[𝐱,k]'Q[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10,j=1:10]
            @test (curl(Q)'curl(Q))[1:10,1:10] ≈ [sum(curl(Q)[𝐱,k]'curl(Q)[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10, j=1:10]
        end

        @testset "k = 2" begin
            n = 30
            @testset "Natural (Dirchlet) m = $m" for m = -4:4
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
            @testset "Neumann  m = $m" for m = -4:4
                P = DiskBasis(m)
                N = Weighted(DiskVector(m, (0,1)))
                JR = Block.(1:n)
                KR = Block.(1:(n-!iszero(m)))

                M_N = (N'N)[JR,JR]
                D_N = Matrix((P'curl(N))[1:n,JR])
                M_P = (P'P)[1:n,1:n]
                Z = zero(M_P)
                A_N = [-Matrix(M_N) D_N'; D_N Z]
                Z₂ = zero(M_N)
                Z₃ = zero(D_N)
                B = [Z₂ Z₃'; Z₃ M_P]
                λ_N = eigvals(A_N, B)
                @test filter(x -> real(x)>0, λ_N)[1:3] ≈ findall(iszero, diff([besselj(abs(m),r) for r in 0.01..13]))[1:3].^2
            end
        end
    end

    @testset "Cylinder" begin
        @testset "Weak form" begin
            @testset "Dirichlet m = $m, k = $k" for m = -2:2, k = -2:2
                W = Weighted(DiskBasis(m, 1))
                M = 2π * (W'W)
                Δ = 2π * grad(W)'grad(W) + k^2 * M

                # We can compute the low order eigenvalues/functions to high accuracy:
                n = 20
                λ,Q = eigen(Symmetric(Δ[1:n,1:n]), Symmetric(M[1:n,1:n]))

                # the eigenvalues are shifted:
                @test λ[1:4] ≈ findall(iszero, [besselj(m,x) for x in 1..20])[1:4] .^ 2 .+ k^2
            end

            @testset "Neumann m = $m, k = $k" for m = -2:2, k = -2:2
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


            C = 2π * [Matrix(ω^2*M_N+C_N)    -im*ω*D          ;
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
