using DeRhamOrthogonalPolynomials, ForwardDiff, Test
using ContinuumArrays: basis

function harmonicpart(f)
    c = transform(Laurent(), θ -> f[RadialCoordinate(1,θ)])
    splat((x,y) -> sum(c[k] * (isodd(k) ? (x+im*y)^(k÷2) : (x-im*y)^(k÷2))  for k in colsupport(c)))
end

@testset "Disk" begin
    @testset "DiskBasis" begin
        for m = -5:5
            P = DiskBasis(m)
            @test [sum(P[𝐱,k]'P[𝐱,j] for 𝐱 in UnitDisk()) for k=1:6, j=1:6] ≈ grammatrix(P)[1:6,1:6] ≈ (P'P)[1:6,1:6]
        end

        @testset "comparison with Zernike" begin
            𝐱 = SVector(0.1,0.2)
            @test Zernike()[𝐱,getindex.(Block.(1:2:10),1)] ≈ ComplexZernike()[𝐱,getindex.(Block.(1:2:10),1)] ≈ DiskBasis(0)[𝐱,1:5]

            @test Zernike()[𝐱,getindex.(Block.(2:2:10),1)] ≈ (DiskBasis(1)[𝐱,1:5] - DiskBasis(-1)[𝐱,1:5])/2im
            @test Zernike()[𝐱,getindex.(Block.(2:2:10),2)] ≈ (DiskBasis(1)[𝐱,1:5] + DiskBasis(-1)[𝐱,1:5])/2

            @test ComplexZernike()[𝐱,getindex.(Block.(2:2:10),1)] ≈ DiskBasis(-1)[𝐱,1:5]
            @test ComplexZernike()[𝐱,getindex.(Block.(2:2:10),2)] ≈ DiskBasis(1)[𝐱,1:5]
        end
    end

    @testset "bubbles" begin
        𝐱 = SVector(0.1,0.2)
        r,θ = norm(𝐱), atan(reverse(𝐱)...)
        for m = -5:5
            W = Weighted(DiskBasis(m, 1))
            @test eltype(W) == ComplexF64
            @test W[𝐱,1:5] ≈ (1-r^2) * r^abs(m) * exp(im*m*θ) * jacobip.(0:4, 1, abs(m), 2r^2-1)
            @test isapproxdiagonal([sum(expand(Zernike(), 𝐱 -> ((conj(W[𝐱, k]) *  W[𝐱, j] / (1-norm(𝐱)^2))))) for k=1:5, j=1:5])
        end
    end

    @testset "vector" begin
        θ= 0.1; ρ = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        𝐱 = SVector(0.1,0.2)

        for m = -4:4
            P = DiskVector(m)
            if m == 0
                @test P[SVector(0.1,0),1] ≈ [0.1,0]
                @test P[SVector(0.1,0),2] ≈ [0,0.1]
            elseif m == 1
                @test P[SVector(0.1,0),1] ≈ [1,im]
            elseif m == -1
                @test P[SVector(0.1,0),1] ≈ [1,-im]
            end

            M = P'P
            @test M isa Diagonal
            @test M[1:6,1:6] ≈ [sum(P[𝐱,k]'P[𝐱,j] for 𝐱 in UnitDisk()) for k=1:6, j=1:6]

            for k = 1:5
                @test P[ρ * 𝐱, k] ≈ ρ * P[𝐱, k] * exp(im*m*θ)
            end
        end
    end

    @testset "matrix" begin
        φ = 0.1; ρ = [cos(φ) -sin(φ); sin(φ) cos(φ)]
        𝐱 = (x,y) = SVector(0.1,0.2)
        r,θ = norm(𝐱), atan(y,x)
        Σ = [x^2-y^2 2x*y; 2x*y y^2-x^2]

        @testset "degree 0 and 1" begin
            @test DiskMatrix(2)[𝐱,1] ≈ [1 im; im -1]
            @test DiskMatrix(-2)[𝐱,1] ≈ [1 -im; -im -1]
            @test DiskMatrix(0)[𝐱,1] ≈ [1 im; -im 1]
            @test DiskMatrix(0)[𝐱,2] ≈ [1 -im; im 1]
            for s in (1,-1)
                @test DiskMatrix(3s)[𝐱,1] ≈ [1 s*im; s*im -1] * (x+s*im*y)
                @test DiskMatrix(s)[𝐱,1] ≈ [1 s*im; s*im -1] * (x-s*im*y)
                @test DiskMatrix(s)[𝐱,2] ≈ [1 s*im; -s*im 1] * (x+s*im*y)
                @test DiskMatrix(s)[𝐱,3] ≈ [1 -s*im; s*im 1] * (x+s*im*y)
            end
        end

        @testset "blocks" begin
            @test DiskMatrix(0)[𝐱,Block(1)] == DiskMatrix(0)[𝐱,1:2]
            @test DiskMatrix(0)[𝐱,Block(2)] == DiskMatrix(0)[𝐱,3:6]
            @test DiskMatrix(1)[𝐱,Block(1)] == DiskMatrix(1)[𝐱,1:3]
            @test DiskMatrix(1)[𝐱,Block(2)] == DiskMatrix(1)[𝐱,4:7]
            @test DiskMatrix(2)[𝐱,Block(1)] == DiskMatrix(2)[𝐱,1:1]
            @test DiskMatrix(2)[𝐱,Block(2)] == DiskMatrix(2)[𝐱,2:4]
            @test DiskMatrix(2)[𝐱,Block(3)] == DiskMatrix(2)[𝐱,5:8]
            @test DiskMatrix(2) == DiskMatrix(2,0) ≠ DiskMatrix(2,1)
            @test eltype(DiskMatrix(2)) == SMatrix{2,2,ComplexF64,4}
        end

        for λ = 0:2, m = -4:4
            P = DiskMatrix(m, λ)
            s = m ≥ 0 ? 1 : -1
            Y = r^(abs(m)-2) * exp(im*(m-2s)*θ) * [1 s*im; s*im -1]
            τ = 2r^2-1

            @testset "definition" begin
                for j = 2:4
                    Z = P[𝐱,Block(j + (abs(m) ≥ 2))]
                    if abs(m) ≥ 2
                        @test Z[1] ≈ jacobip(j, λ, abs(m)-2, τ) * Y
                    elseif abs(m) == 1
                        @test Z[1] ≈ (1+τ) * (j+λ)/2j * jacobip(j-1, λ, 1, τ) * Y
                    else
                        @test Z[1] ≈ (1+τ)^2 * (j+λ-1)*(j+λ)/(4j*(j-1)) * jacobip(j-2, λ, 2, τ) * Y
                    end
                    @test Z[2] ≈ jacobip(j-1, λ, abs(m), τ) * Σ * Y
                    @test Z[3] ≈ jacobip(j-1, λ, abs(m), τ) * Y * Σ
                    @test Z[4] ≈ jacobip(j-2, λ, abs(m)+2, τ) * Σ * Y * Σ
                end
            end

            @testset "symmetry-adapted" begin
                for k = 1:10
                    @test P[ρ * 𝐱, k] * ρ ≈ ρ * P[𝐱, k] * exp(im*m*φ)
                end
            end

            @testset "orthogonality" begin
                M = [sum(tr(P[𝐱,k]'P[𝐱,j]) * (1-norm(𝐱)^2)^λ for 𝐱 in UnitDisk()) for k=1:10, j=1:10]
                @test isapproxdiagonal(M)
                @test all(>(0.001), real(diag(M)))
            end
        end
    end

    @testset "normal" begin
        𝐱 = (x,y) = SVector(0.1,0.2)

        @testset "orthogonality" begin
            for m = -4:4
                P = DiskVector(m)
                Q = DiskVector(m, (0,1))
                K = Weighted(DiskCurlFree(m))
                W = Weighted(Q)
                if m == 1
                    @test Q[SVector(0.1,0),1] ≈ [1/2,im/2]
                elseif m == -1
                    @test Q[SVector(0.1,0),1] ≈ [1/2,-im/2]
                end

                if iszero(m)
                    @test K[𝐱,1:5] ≈ [2W[𝐱,Block(k)[1]] for k = 1:5]
                else
                    @test K[𝐱,1:5] ≈ [sum(W[𝐱,Block(k)]) for k = 2:6]
                end

                @test (Q\P)[1:10,1:10]'Q[𝐱,1:10] ≈ P[𝐱,1:10]
                @test (P\W)[1:13,1:10]'P[𝐱,1:13] ≈ W[𝐱,1:10]
                @test (W'W)[1:6,1:6] ≈ [sum(W[𝐱,k]'W[𝐱,j] for 𝐱 in UnitDisk()) for k=1:6, j=1:6]

                @test (W\K)[1:13,1:6]'W[𝐱,1:13] ≈ K[𝐱,1:6]
                @test (P\K)[1:13,1:6]'P[𝐱,1:13] ≈ K[𝐱,1:6]
                @test isdiag((K'K)[1:10,1:10])
                @test (K'K)[1:10,1:10] ≈ [sum(K[𝐱,k]'K[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10, j=1:10]

                @test isapproxdiagonal([sum(
                        ((x,y) = 𝐱;
                        Q[𝐱,k]' * [1-y^2 x*y; x*y 1-x^2] * Q[𝐱,j])
                    for 𝐱 in UnitDisk()) for k=1:6, j=1:6])


                θ= 0.1; ρ = [cos(θ) -sin(θ); sin(θ) cos(θ)]
                x,y = 𝐱;
                for k = 1:5
                    @test Q[ρ * 𝐱, k] ≈ ρ * Q[𝐱, k] * exp(im*m*θ)
                    @test W[𝐱,k] ≈ [1-y^2 x*y; x*y  1-x^2] * Q[𝐱,k]
                end
            end
        end

        @testset "raising/lowering" begin
            for m = -5:5
                P = DiskVector(m)
                Q = DiskVector(m, (0,1))
                N = Weighted(Q)
                K = Weighted(DiskCurlFree(m))
                W = Weighted(DiskBasis(m, 1))

                L = P \ N
                @test L[1:13,1:10]'P[𝐱,1:13] ≈ N[𝐱,1:10]

                R = Q \ P
                @test P[𝐱,1:13] ≈ R[1:13,1:13]'Q[𝐱,1:13]

                for 𝐚 in ((-1,0), (0,-1), (0,1), (1,0))
                    S, T = DiskVector(m, 𝐚), DiskVector(m, 𝐚 .+ 1)
                    @test S[𝐱,1:13] ≈ (T \ S)[1:13,1:13]'T[𝐱,1:13]
                end
                for 𝐚 in ((-1,0), (0,-1), (1,2), (2,1))
                    S, T = DiskVector(m, (min(𝐚...), min(𝐚...))), DiskVector(m, 𝐚)
                    @test S[𝐱,1:13] ≈ (T \ S)[1:13,1:13]'T[𝐱,1:13]
                end

                @test (N \ K)[1:11,1:5]'N[𝐱,1:11] ≈ K[𝐱,1:5]
                @test (P \ K)[1:11,1:5]'P[𝐱,1:11] ≈ K[𝐱,1:5]
                @test  (P \ K)[1:11,1:5] ≈ (P \ N)[1:11,1:11] * (N \K)[1:11,1:5]

                @test (K'N)[1:10,1:10] ≈ (N'K)[1:10,1:10]' ≈ [sum(K[𝐱,k]'N[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10,j=1:10] 
                @test (N'grad(W))[1:10,1:10] ≈ (grad(W)'N)[1:10,1:10]' ≈ [sum(N[𝐱,k]'grad(W)[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10,j=1:10]
            end
        end

        @testset "equivariantJacobi" begin
            for m = -5:5
                P = DiskVector(m)
                Q = DiskVector(m, (0,1))
                N = Weighted(Q)
                A,B = equivariantjacobimatrix(Val(1), P), equivariantjacobimatrix(Val(2), P);

                r = norm(𝐱)
                x,y = 𝐱
                @test A[1:12,1:10]'*P[𝐱, 1:12] ≈ r^2*P[𝐱, 1:10]
                @test B[1:13,1:10]'*P[𝐱, 1:13] ≈ Ref([x^2-y^2 2x*y; 2x*y y^2-x^2]) .* P[𝐱, 1:10]
                # N == [1-y^2 x*y; x*y 1-x^2] ==  (1-(x^2+y^2)/2)*I + G/2
                L = P \ N
                R = Q \ P
                @test (L*R)[1:10,1:10] ≈ (I + (B-A)/2)[1:10,1:10]
            end
        end

        @testset "WeightedGram" begin
            for m = -3:3
                N = DiskVector(m, (0,1))
                K = DiskCurlFree(m)
                C = DiskCoCurlFree(m)
                @test [sum(Weighted(K)[𝐱,k]'K[𝐱,j] for 𝐱 in UnitDisk()) for k=1:5, j=1:5] ≈ (K'Weighted(K))[1:5,1:5]
                @test [sum(Weighted(C)[𝐱,k]'C[𝐱,j] for 𝐱 in UnitDisk()) for k=1:5, j=1:5] ≈ (C'Weighted(C))[1:5,1:5]
                @test [sum(Weighted(N)[𝐱,k]'N[𝐱,j] for 𝐱 in UnitDisk()) for k=1:5, j=1:5] ≈ (N'Weighted(N))[1:5,1:5]
                
                if m ≠ 0
                    @test N[𝐱,1] ≈ -C[𝐱,1]/(2abs(m))
                    for j = 1:5
                        @test N[𝐱,Block(j+1)[1]] ≈ (4*(2j+abs(m)+1)*(j+abs(m)))*(K[𝐱,j] + C[𝐱,j+1]/(2*(j+abs(m))))/ 𝔯(j, abs(m)-1)
                        @test N[𝐱,Block(j+1)[2]] ≈ (j*(2j+abs(m)+2))*(K[𝐱,j] - C[𝐱,j+1]*2*(2j+abs(m)+1)/(j*(2j+abs(m)+2)))/ 𝔯(j, abs(m)-1)
                    end
                end
                # n^+ + n^-
                # -C[𝐱,1]/2
            end
        end
    end

    @testset "gradient" begin
        @testset "Bubbles" begin
            𝐱 = SVector(0.1,0.2)
            for m = -5:5
                W = Weighted(DiskBasis(m, 1))
                N = Weighted(DiskVector(m, (0,1)))
                ∇ = grad(W)
                for n = 1:10
                    @test gradient(𝐱 -> Weighted(DiskBasis{eltype(𝐱)}(m,1))[𝐱,n], 𝐱) ≈ ∇[𝐱,n]
                    if iszero(m)
                    @test -2n*N[𝐱,Block(n)[1]] ≈ ∇[𝐱,n]
                    else
                        @test -n*sum(N[𝐱,Block(n+1)]) ≈ ∇[𝐱,n]
                    end
                end

                @test ∇'∇ isa Diagonal
                @test (∇'∇)[1:10,1:10] ≈ [sum(∇[𝐱,k]'∇[𝐱,j] for 𝐱 in UnitDisk()) for k=1:10, j=1:10]
            end
        end
    end


    @testset "curl" begin
        @testset "m" begin
            𝐱 = SVector(0.1,0.2)
            for m = -5:5
                iszero(m) && continue
                P = DiskBasis(m)
                N = Weighted(DiskVector(m, (0,1)))
                C = curl(N)

                𝔯 = (n,b) -> 10n^2+n*(13b+19)+4*(b+1)*(b+2)

                if iszero(m)
                    for j = 1:2:10
                        @test fd_curl(𝐱 -> Weighted(DiskVector{eltype(𝐱)}(m, (0,1)))[𝐱,j], 𝐱) ≈ C[𝐱,j] atol=1E-10
                        @test C[𝐱,j] ≈ 0 atol=1E-10
                    end
                    for j = 2:2:10
                        @test fd_curl(𝐱 -> Weighted(DiskVector{eltype(𝐱)}(m, (0,1)))[𝐱,j], 𝐱) ≈ C[𝐱,j] ≈ -2*(j÷2) *DiskBasis(0)[𝐱,j÷2+1]
                    end
                else
                    @test fd_curl(𝐱 -> Weighted(DiskVector{eltype(𝐱)}(m, (0,1)))[𝐱,1], 𝐱) ≈  -2*abs(m)*(abs(m)+1)*(abs(m)+2)/𝔯(0,abs(m)-1) * im*sign(m) *  P[𝐱,1] ≈ C[𝐱,1]
                    for j = 1:10
                        @test fd_curl(𝐱 -> Weighted(DiskVector{eltype(𝐱)}(m, (0,1)))[𝐱,Block(j+1)[1]], 𝐱) ≈
                            2*(2j+abs(m))*(2j+abs(m)+1)*(2j+abs(m)+2)/𝔯(j,abs(m)-1) * im*sign(m)*  P[𝐱,j+1] ≈ C[𝐱,Block(j+1)[1]]
                        @test fd_curl(𝐱 -> Weighted(DiskVector{eltype(𝐱)}(m, (0,1)))[𝐱,Block(j+1)[2]], 𝐱) ≈
                            -2*(2j+abs(m))*(2j+abs(m)+1)*(2j+abs(m)+2)/𝔯(j,abs(m)-1) * im*sign(m)*  P[𝐱,j+1] ≈ C[𝐱,Block(j+1)[2]]
                    end
                end

            end
        end
    end

    @testset "divergence" begin
        𝐱 = SVector(0.1,0.2)
        for m = -2:2, a = -1:0
            D = divergence(DiskVector(m, (a,a+1)))
            @test basis(D) == DiskBasis(m, a+1)
            for n = 1:10
                @test fd_div(𝐱 -> DiskVector{eltype(𝐱)}(m, (a,a+1))[𝐱,n], 𝐱) ≈ D[𝐱,n] atol=1E-10
            end
        end
        @test_throws AssertionError divergence(DiskVector(1, (1,0)))
    end

    @testset "inversecurl" begin
        𝐱 = SVector(0.1,0.2)
        for m = -3:3
            P = DiskBasis(m)
            Q = DiskVector(m, (0,1))
            if m ≠ 0
                for j = 0:3
                    @test [0 -1; 1 0] * gradient(𝐱 -> DiskBasis{eltype(𝐱)}(m)[𝐱,j+1], 𝐱) ≈ -2im*sign(m)*(j+abs(m))*Q[𝐱, Block(j+1)[2]] + im*sign(m)*j*(2+abs(m)+2j)/(2*(2j+abs(m)+1)) * Q[𝐱, Block(j+1)[1]]
                end
            end
            @test rotgrad(P)[𝐱,1:10] ≈ [[0 -1; 1 0] * gradient(𝐱 -> DiskBasis{eltype(𝐱)}(m)[𝐱,j], 𝐱) for j=1:10]
            
            V = DiskVector(m)
            D = Weighted(DiskCoCurlFree(m))
            N = Weighted(DiskVector(m, (0,1)))

            let n = 10
                @test (N\D)[1:2n+1,1:n]'N[𝐱,1:2n+1] ≈ D[𝐱,1:n]
                @test (V\N)[1:2n+3,1:2n]'V[𝐱,1:2n+3] ≈ N[𝐱,1:2n]
                @test (V\D)[1:2n+3,1:n]'V[𝐱,1:2n+3] ≈ D[𝐱,1:n]
            end

            @test norm([sum(DiskCurlFree(m)[𝐱,k]'DiskVectorWeight((0,1))[𝐱]DiskCoCurlFree(m)[𝐱,j] for 𝐱 in UnitDisk()) for k=1:5, j=1:5]) ≤ 1E-10
            @test isapproxdiagonal([sum(DiskCoCurlFree(m)[𝐱,k]'DiskVectorWeight((0,1))[𝐱]DiskCoCurlFree(m)[𝐱,j] for 𝐱 in UnitDisk()) for k=1:5, j=1:5])
        
            @test [sum(DiskCoCurlFree(m)[𝐱,k]'DiskVectorWeight((0,1))[𝐱]^2*DiskCoCurlFree(m)[𝐱,j] for 𝐱 in UnitDisk()) for k=1:5, j=1:5] ≈ (D'D)[1:5,1:5]


            @test [fd_curl(𝐱 -> Weighted(DiskCoCurlFree{eltype(𝐱)}(m))[𝐱,j], 𝐱) for j = 1:10] ≈ curl(D)[𝐱,1:10]
        end
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
end
