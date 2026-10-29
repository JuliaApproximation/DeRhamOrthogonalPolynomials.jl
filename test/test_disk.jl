using DeRhamOrthogonalPolynomials, ForwardDiff, Test

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

    include("test_generalised_disk.jl")
end
