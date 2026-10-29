using DeRhamOrthogonalPolynomials, ClassicalOrthogonalPolynomials, BlockArrays, Test, FastGaussQuadrature, StaticArrays, LinearAlgebra, DomainSets
using ForwardDiff
using ClassicalOrthogonalPolynomials: jacobimatrix, weightedgrammatrix, HalfWeighted
using DeRhamOrthogonalPolynomials: 𝔯, RadialCoordinate, laddergrad, laddercurl

# isapproxdiagonal(A) = A ≈ Diagonal(diag(A))

_𝔯_am1(n, b) = 10n^2 + (13b+7)*n + 4b*(b+1)
function _w_vec(n, b, t)
    τ = 2t-1
    p1 = jacobip(n, 0, b+1, τ)
    p2n = jacobip(n, 0, b+2, τ)
    p2m = n ≥ 1 ? jacobip(n-1, 0, b+2, τ) : zero(t)
    c = 2n*(n+b+2)/(2n+b+2)
    [(b+1)*(1-t)*p1 - c*t*(p2n - p2m), (b+1)*p1]
end
function _q1_formula(n, b, t)
    τ = 2t-1
    r = _𝔯_am1(n, b)
    (2n+b)/(n*r) * ((2n+b+1)^2 * jacobip(n, 0, b, τ) * [(1-t), 1.0] - (n+b+1)*_w_vec(n, b, t))
end
_q2_formula(n, b, t) = -(2n+b)/(n*(2n+b-1)) * _w_vec(n-1, b, t) - _q1_formula(n, b, t)
_𝔯_a0(n, b) = 10n^2 + (13b+19)*n + 4*(b+1)*(b+2)
function _v_vec_a0(n, b, t)
    τ = 2t-1
    [(b+1)*(1-t)*jacobip(n,1,b+1,τ) - 2t*(n+1)*jacobip(n,0,b+2,τ), (b+1)*jacobip(n,1,b+1,τ)]
end
function _q1_a0(n, b, t)
    τ = 2t-1
    r = _𝔯_a0(n, b)
    (2n+b+1)/((n+1)*r) * ((2n+b+2)*(2n+b+3)*jacobip(n,1,b,τ)*[(1-t), 1.0] - (n+b+2)*_v_vec_a0(n,b,t))
end
_q2_a0(n, b, t) = -_v_vec_a0(n-1, b, t)/n - _q1_a0(n, b, t)

@testset "VectorJacobi" begin
    @testset "Diagonal weight" begin
        @testset "evaluation and gram matrix" begin
            for a = 0:5, b = 0:5
                P = VectorJacobi((a,a), b)
                t = 0.1
                @test P[t,1:2:5] ≈ vectorjacobip.(0:2, 0, Ref((a,a)), b, t) ≈ Ref([1,1]) .* jacobip.(0:2, a, b, 2t-1)
                @test P[t,2:2:6] ≈ vectorjacobip.(1:3, 1, Ref((a,a)), b, t) ≈ Ref([1,-1]) .* (t * jacobip.(0:2, a, b+2, 2t-1))

                W = Weighted(P)
                @test W[t,1:10] ≈ (1-t)^a * t^b * P[t,1:10]

                G = [sum(P[s,k]'P[s,j]*(1-s)^a*s^b for s in 0..1) for k=1:10, j=1:10]
                @test isapproxdiagonal(G)
                @test G[1:2:end,1:2:end] ≈ weightedgrammatrix(Jacobi(a,b))[1:5,1:5]/2^(a+b)
                @test G[2:2:end,2:2:end] ≈ weightedgrammatrix(Jacobi(a,b+2))[1:5,1:5]/2^(a+b+2)
            end
        end

        @testset "jacobimatrix" begin
            for b = 0:5
                t = 0.1
                P = VectorJacobi((0,0), b)
                A,B = jacobimatrix(Val(1), P),jacobimatrix(Val(2), P)
                @test t * P[t,1:10] ≈ A[1:20,1:10]'P[t,1:20]
                @test Ref([t 0; 0 -t]) .* P[t,1:10] ≈ B[1:20,1:10]'P[t,1:20]
            end
        end
    end

    @testset "Normal" begin
        @testset "evaluation" begin
            t = 0.1
            for b = 0:5
                P = VectorJacobi((0,0), b)
                Q = VectorJacobi((0,1), b)
                W = Weighted(Q)
                w = VectorJacobiWeight((0,1),b)
                @test w[t] ≈ Diagonal([1,1-t]) * t^b
                @test W[t,1:10] ≈ Ref(w[t]) .* Q[t,1:10]
            end
        end

        @testset "conversion" begin
            for b = 0:5
                P = VectorJacobi((0,0), b)
                Q = VectorJacobi((0,1), b)
                W = Weighted(Q)
                V = Weighted(P)
                w = VectorJacobiWeight((0,1), b)
                L = V\W
                R = Q\P
                A,B = jacobimatrix(Val(1), P),jacobimatrix(Val(2), P);
                t = 0.1
                @test (P * (I + (B-A)/2))[t,1:10] * t^b ≈ Ref(w[t]) .* P[t,1:10]
                @test (L*R)[1:10,1:10] ≈ (I + (B-A)/2)[1:10,1:10]
                @test (V * L)[t,1:10] ≈ W[t,1:10]
                @test (Q * R)[t,1:10] ≈ P[t,1:10]
            end
        end
    end

    @testset "General a" begin
        for a = 0:5, b = 0:5
            Q = VectorJacobi((a,a+1), b)
            @test isapproxdiagonal([sum(Q[t,k]' * Diagonal([(1-t)^a, (1-t)^(a+1)]) * Q[t,j] * t^b for t in 0..1) for k=1:10, j=1:10])
        end

        a = -1
        for b = 0:5
            Q = VectorJacobi((a,a+1), b)
            ip = (𝐟, 𝐠) -> sum((𝐟[t]-𝐟[1.0][1]*[1,1])' * Diagonal([(1-t)^a, (1-t)^(a+1)]) * (𝐠[t]-𝐠[1.0][1]*[1,1]) * t^b for t in 0..1) + 𝐟[1.0][1]*𝐠[1.0][1]
            @test isapproxdiagonal([ip(Q[:,k], Q[:,j]) for k=1:5, j=1:5])
        end
    end

    @testset "Reversed (a+1,a)" begin
        @testset "component swap" begin
            t = 0.1
            for a = -1:4, b = 0:4
                P = VectorJacobi((a, a+1), b)
                Q = VectorJacobi((a+1, a), b)
                for k = 1:10
                    pv = P[t, k]
                    @test Q[t, k] ≈ [pv[2], pv[1]]
                end
            end
        end

        @testset "orthogonality" begin
            for a = 0:4, b = 0:4
                Q = VectorJacobi((a+1, a), b)
                G = [sum(Q[t,k]' * Diagonal([(1-t)^(a+1), (1-t)^a]) * Q[t,j] * t^b for t in 0..1) for k=1:10, j=1:10]
                @test isapproxdiagonal(G)
            end
        end
    end

    @testset "(1,0): reversed normal at a=0" begin
        @testset "component swap" begin
            t = 0.1
            for b = 0:5
                P = VectorJacobi((0, 1), b)
                Q = VectorJacobi((1, 0), b)
                for k = 1:10
                    pv = P[t, k]
                    @test Q[t, k] ≈ [pv[2], pv[1]]
                end
            end
        end

        @testset "n=0 formula" begin
            # The zeroth OP is [1/2,1/2] regardless of t and b
            for b = 0:5
                Q = VectorJacobi((1, 0), b)
                for t in [0.1, 0.4, 0.9]
                    @test Q[t, 1] ≈ [0.5, 0.5]
                end
            end
        end

        @testset "orthogonality" begin
            for b = 0:5
                Q = VectorJacobi((1, 0), b)
                G = [sum(Q[t,k]' * Diagonal([1-t, 1]) * Q[t,j] * t^b for t in 0..1) for k=1:10, j=1:10]
                @test isapproxdiagonal(G)
            end
        end
    end

    @testset "(0,-1): singular reversed weight" begin
        @testset "component swap vs (-1,0)" begin
            t = 0.1
            for b = 0:5
                P = VectorJacobi((-1, 0), b)
                Q = VectorJacobi((0, -1), b)
                for k = 1:10
                    pv = P[t, k]
                    @test Q[t, k] ≈ [pv[2], pv[1]]
                end
            end
        end

        @testset "n=0 formula" begin
            for b = 0:5
                Q = VectorJacobi((0, -1), b)
                for t in [0.1, 0.5, 0.9]
                    @test Q[t, 1] ≈ [0.5, 0.5]
                end
            end
        end

        @testset "regularized orthogonality" begin
            # Regularized IP: subtract f_2(1)*[1,1] (second component at t=1) from both
            # components before integrating against diag(1,(1-t)^{-1})*t^b; add boundary term
            for b = 0:5
                Q = VectorJacobi((0, -1), b)
                ip = (𝐟, 𝐠) -> sum((𝐟[t] - 𝐟[1.0][2]*[1,1])' * Diagonal([1, 1/(1-t)]) * (𝐠[t] - 𝐠[1][2]*[1,1]) * t^b for t in 0..1) + 𝐟[1][2]*𝐠[1][2]
                @test isapproxdiagonal([ip(Q[:,k], Q[:,j]) for k=1:8, j=1:8])
            end
        end
    end

    @testset "n=0 OP equals [1/2,1/2] for normal and reversed families" begin
        for t in [0.0, 0.3, 0.7, 1.0], b = 0:4, a = -1:4
            @test VectorJacobi((a, a+1), b)[t, 1] ≈ [0.5, 0.5]
            @test VectorJacobi((a+1, a), b)[t, 1] ≈ [0.5, 0.5]
        end
    end

    @testset "module: equal components at t=0" begin
        # Every OP in M satisfies f[1](0) = f[2](0)
        for b = 0:4
            for a = 0:4
                for k = 1:10
                    @test VectorJacobi((a, a), b)[0.0, k][1] ≈ VectorJacobi((a, a), b)[0.0, k][2]
                    @test VectorJacobi((a, a+1), b)[0.0, k][1] ≈ VectorJacobi((a, a+1), b)[0.0, k][2]
                    @test VectorJacobi((a+1, a), b)[0.0, k][1] ≈ VectorJacobi((a+1, a), b)[0.0, k][2]
                end
            end
            # a=-1 singular cases
            for k = 1:10
                @test VectorJacobi((-1, 0), b)[0.0, k][1] ≈ VectorJacobi((-1, 0), b)[0.0, k][2]
                @test VectorJacobi((0, -1), b)[0.0, k][1] ≈ VectorJacobi((0, -1), b)[0.0, k][2]
            end
        end
    end

    @testset "a=-1 to a=0 conversion formulas" begin
        # Scalar Jacobi connection: P_n^{(-1,b)} = (n+b)/(2n+b) * (P_n^{(0,b)} - P_{n-1}^{(0,b)})
        @testset "scalar Jacobi connection" begin
            for n in 1:5, b in 0:4, t in (0.1, 0.4, 0.7)
                lhs = jacobip(n, -1, b, 2t-1)
                rhs = (n+b)/(2n+b) * (jacobip(n, 0, b, 2t-1) - jacobip(n-1, 0, b, 2t-1))
                @test lhs ≈ rhs
            end
        end

        # Diagonal basis: p^{ν,(-1,b)}_n in terms of p^{ν,(0,b)}_n via the scalar connection above.
        @testset "diagonal basis conversion" begin
            for b in 0:4, t in (0.2, 0.5, 0.8)
                Pm1 = VectorJacobi((-1,-1), b)
                P0  = VectorJacobi((0,0), b)
                # type-1: p^{1,(-1,b)}_n = (n+b)/(2n+b) * (p^{1,(0,b)}_n - p^{1,(0,b)}_{n-1})
                for n in 1:5
                    lhs  = Pm1[t, 2n+1]
                    prev = n == 1 ? P0[t, 1] : P0[t, 2(n-1)+1]
                    rhs  = (n+b)/(2n+b) * (P0[t, 2n+1] - prev)
                    @test lhs ≈ rhs atol=1e-12
                end
                # type-2: p^{2,(-1,b)}_1 = p^{2,(0,b)}_1
                @test Pm1[t, 2] ≈ P0[t, 2] atol=1e-12
                # type-2: p^{2,(-1,b)}_n = (n+b+1)/(2n+b) * (p^{2,(0,b)}_n - p^{2,(0,b)}_{n-1}) for n≥2
                for n in 2:5
                    lhs = Pm1[t, 2n]
                    rhs = (n+b+1)/(2n+b) * (P0[t, 2n] - P0[t, 2(n-1)])
                    @test lhs ≈ rhs atol=1e-12
                end
            end
        end

        @testset "normal basis a=-1 explicit formula" begin
            for b in 0:4
                Q = VectorJacobi((-1, 0), b)
                for n in 1:5, t in (0.1, 0.4, 0.7)
                    # Linear index 2n+1 = type-1, 2n = type-2 at degree n
                    @test Q[t, 2n+1] ≈ _q1_formula(n, b, t) atol=1e-12
                    @test Q[t, 2n]   ≈ _q2_formula(n, b, t) atol=1e-12
                end
            end
        end

        @testset "normal basis a=0 explicit formula" begin
            for b in 0:4
                Q = VectorJacobi((0, 1), b)
                for n in 1:5, t in (0.1, 0.4, 0.7)
                    @test Q[t, 2n+1] ≈ _q1_a0(n, b, t) atol=1e-12
                    @test Q[t, 2n]   ≈ _q2_a0(n, b, t) atol=1e-12
                end
            end
        end

        @testset "normal basis conversion a=-1 → a=0" begin
            # α=0→1 connection: P_n^(0,β) = (n+β+1)/(2n+β+1)*P_n^(1,β) - (n+β)/(2n+β+1)*P_{n-1}^(1,β)
            @testset "alpha=0 to 1 Jacobi shift" begin
                for n in 1:6, β in 0:4, t in (0.15, 0.45, 0.75)
                    τ = 2t-1
                    lhs = jacobip(n, 0, β, τ)
                    rhs = (n+β+1)/(2n+β+1)*jacobip(n, 1, β, τ) - (n+β)/(2n+β+1)*jacobip(n-1, 1, β, τ)
                    @test lhs ≈ rhs atol=1e-12
                end
            end

            # ζ_n = (n+b+1)/(2n+b+1): coefficient of q_{n,1}^{(0,b)} in q_{n,1}^{(-1,b)}
            # and α_n = -(n-1)(n+b+1)/(n(2n+b-1)): coefficient of q_{n-1,2}^{(0,b)} in q_{n,2}^{(-1,b)}
            @testset "diagonal coefficients zeta_n and alpha_n" begin
                t1, t2, t3 = 0.2, 0.5, 0.8
                for b in 0:4, n in 2:5
                    Q0  = VectorJacobi((0,1), b)
                    Qm1 = VectorJacobi((-1,0), b)
                    ζ = (n+b+1)/(2n+b+1)
                    α = -(n-1)*(n+b+1)/(n*(2n+b-1))

                    # Type-1: residual r(t) = q_{n,1}^{(-1,b)}(t) - ζ * q_{n,1}^{(0,b)}(t)
                    # should lie in span{q_{n-1,1}^{(0,b)}, q_{n,2}^{(0,b)}} (two basis vectors, 2 components)
                    # Solve for coefficients at t1,t2 and verify at t3
                    for (t_solve1, t_solve2, t_check) in ((t1,t2,t3), (t2,t3,t1))
                        # residual at three points
                        r1 = Qm1[t_solve1, 2n+1] - ζ * Q0[t_solve1, 2n+1]
                        r2 = Qm1[t_solve2, 2n+1] - ζ * Q0[t_solve2, 2n+1]
                        r3 = Qm1[t_check,  2n+1] - ζ * Q0[t_check,  2n+1]
                        # Basis vectors at each t
                        B1_1 = Q0[t_solve1, 2n-1];  B2_1 = Q0[t_solve1, 2n]
                        B1_2 = Q0[t_solve2, 2n-1];  B2_2 = Q0[t_solve2, 2n]
                        B1_3 = Q0[t_check,  2n-1];  B2_3 = Q0[t_check,  2n]
                        # 4×2 system [B1(t_i) | B2(t_i)] * [δ; ε] = r(t_i) for i=1,2
                        A_sys = [B1_1[1] B2_1[1]; B1_1[2] B2_1[2]; B1_2[1] B2_2[1]; B1_2[2] B2_2[2]]
                        b_sys = [r1[1]; r1[2]; r2[1]; r2[2]]
                        coeff = A_sys \ b_sys
                        δ_est, ε_est = coeff[1], coeff[2]
                        # Verify at t3
                        pred = δ_est * B1_3 + ε_est * B2_3
                        @test r3 ≈ pred atol=1e-9
                    end

                    # Type-2: residual r(t) = q_{n,2}^{(-1,b)}(t) - α * q_{n-1,2}^{(0,b)}(t)
                    # should lie in span{q_{n-1,1}^{(0,b)}, q_{n,2}^{(0,b)}}
                    for (t_solve1, t_solve2, t_check) in ((t1,t2,t3), (t2,t3,t1))
                        r1 = Qm1[t_solve1, 2n]   - α * Q0[t_solve1, 2n-2]
                        r2 = Qm1[t_solve2, 2n]   - α * Q0[t_solve2, 2n-2]
                        r3 = Qm1[t_check,  2n]   - α * Q0[t_check,  2n-2]
                        B1_1 = Q0[t_solve1, 2n-1];  B2_1 = Q0[t_solve1, 2n]
                        B1_2 = Q0[t_solve2, 2n-1];  B2_2 = Q0[t_solve2, 2n]
                        B1_3 = Q0[t_check,  2n-1];  B2_3 = Q0[t_check,  2n]
                        A_sys = [B1_1[1] B2_1[1]; B1_1[2] B2_1[2]; B1_2[1] B2_2[1]; B1_2[2] B2_2[2]]
                        b_sys = [r1[1]; r1[2]; r2[1]; r2[2]]
                        coeff = A_sys \ b_sys
                        β_est, γ_est = coeff[1], coeff[2]
                        pred = β_est * B1_3 + γ_est * B2_3
                        @test r3 ≈ pred atol=1e-9
                    end
                end
            end

            # Relation γ_n = ζ_n - ε_n: verify by computing both coefficients
            @testset "relation gamma_n = zeta_n - epsilon_n" begin
                t1, t2, t3 = 0.25, 0.55, 0.85
                for b in 0:3, n in 2:4
                    Q0  = VectorJacobi((0,1), b)
                    Qm1 = VectorJacobi((-1,0), b)
                    ζ = (n+b+1)/(2n+b+1)
                    α = -(n-1)*(n+b+1)/(n*(2n+b-1))
                    # Solve for ε_n (coeff of q_{n,2}^{(0,b)} in q_{n,1}^{(-1,b)}) via 2-poitn system
                    r1 = Qm1[t1, 2n+1] - ζ * Q0[t1, 2n+1]
                    r2 = Qm1[t2, 2n+1] - ζ * Q0[t2, 2n+1]
                    A_sys = [Q0[t1,2n-1][1] Q0[t1,2n][1]; Q0[t1,2n-1][2] Q0[t1,2n][2];
                             Q0[t2,2n-1][1] Q0[t2,2n][1]; Q0[t2,2n-1][2] Q0[t2,2n][2]]
                    b_sys = [r1; r2]
                    coeff1 = A_sys \ b_sys; ε_est = coeff1[2]
                    # Solve for γ_n (coeff of q_{n,2}^{(0,b)} in q_{n,2}^{(-1,b)}) via 2-point system
                    s1 = Qm1[t1, 2n] - α * Q0[t1, 2n-2]
                    s2 = Qm1[t2, 2n] - α * Q0[t2, 2n-2]
                    b_sys2 = [s1; s2]
                    coeff2 = A_sys \ b_sys2; γ_est = coeff2[2]
                    @test γ_est ≈ ζ - ε_est atol=1e-8
                end
            end

            @testset "explicit formula for epsilon_n" begin
                # epsilon_n = -2n(n+b+1) / (𝔯_{-1}(n,b) * (2n+b+2))
                𝔯m1(n,b) = 10n^2 + (13b+7)*n + 4b*(b+1)
                t1, t2 = 0.3, 0.7
                for b in 0:5, n in 2:6
                    Q0  = VectorJacobi((0,1), b)
                    Qm1 = VectorJacobi((-1,0), b)
                    ζ = (n+b+1)/(2n+b+1)
                    ε_formula = -2n*(n+b+1) / (𝔯m1(n,b) * (2n+b+2))
                    # Solve for ε numerically via 2-point overdetermined system
                    r1 = Qm1[t1, 2n+1] - ζ * Q0[t1, 2n+1]
                    r2 = Qm1[t2, 2n+1] - ζ * Q0[t2, 2n+1]
                    A_sys = [Q0[t1,2n-1][1] Q0[t1,2n][1]; Q0[t1,2n-1][2] Q0[t1,2n][2];
                             Q0[t2,2n-1][1] Q0[t2,2n][1]; Q0[t2,2n-1][2] Q0[t2,2n][2]]
                    b_sys = [r1; r2]
                    coeff = A_sys \ b_sys; ε_num = coeff[2]
                    @test ε_formula ≈ ε_num atol=1e-9
                end
            end
        end
    end

    @testset "expand" begin
        P = VectorJacobi((0,0), 0)
        Q = VectorJacobi((-1,0), 0)
        @test expand(P, t -> [1+sin(t),exp(t)])[0.1] ≈ expand(Q, t -> [1+sin(t),exp(t)])[0.1] ≈ [1+sin(0.1),exp(0.1)]
    end

    @testset "conversion" begin
        for b = 0:3, a = -1:3
            P = VectorJacobi((a+1,a+1), b)
            for Q in (VectorJacobi((a,a+1), b), VectorJacobi((a+1,a), b))
                R = P\Q
                n = 10
                @test Q[0.1:0.1,1:n] ≈ P[0.1:0.1,1:n] * R[1:n,1:n]
            end
        end

        P = VectorJacobi((0,0), 0)
        for Q in (VectorJacobi((-1,0), 0), VectorJacobi((0,-1), 0))
            R = P\Q
            for j = 1:10
                @test (P\Q[:,j])[1:10] ≈ R[1:10,j]
            end
        end

        @testset "derivation" begin
            t = 0.1
            τ = 2t-1
            a,b = -1,0
            n = 5

            @test VectorJacobi((a+1,a+1), b)[t,Block(n+1)[1]] ≈ t * jacobip(n-1, a+1, b+2, 2t-1) * SVector(1,-1) ≈ (τ+1)/2 * jacobip(n-1, a+1, b+2, τ) * SVector(1,-1) ≈ (jacobip(n, a+1, b+1, τ) * n/(2n+a+b+2) + jacobip(n-1, a+1, b+1, τ) * (n+b+1)/(2n+a+b+2)) * SVector(1,-1)
            @test VectorJacobi((a+1,a+1), b)[t,Block(n+1)[2]] ≈ jacobip(n, a+1, b, τ) * SVector(1,1) ≈ (jacobip(n, a+1, b+1, τ) * (n+a+b+2)/(2n+a+b+2) + jacobip(n-1, a+1, b+1, τ) * n/(2n+a+b+2)) * SVector(1,1)


            @testset "jacobi relationships" begin
                @test jacobip(n,a,b+1,τ) ≈ jacobip(n,a+1,b+1,τ) * (n+a+b+2)/(2n+a+b+2) - jacobip(n-1,a+1,b+1,τ) * (n+b+1)/(2n+a+b+2)
                @test jacobip(n,a+1,b,τ) ≈ jacobip(n,a+1,b+1,τ) * (n+a+b+2)/(2n+a+b+2) + jacobip(n-1,a+1,b+1,τ) * (n+a+1)/(2n+a+b+2)
                @test jacobip(n-1,a+1,b,τ) ≈ jacobip(n-1,a+1,b+1,τ) * (n+a+b+1)/(2n+a+b) + jacobip(n-2,a+1,b+1,τ) * (n+a)/(2n+a+b)
                @test jacobip(n+1,a+1,b,τ) ≈ jacobip(n+1,a+1,b+1,τ) * (n+a+b+3)/(2n+a+b+4) + jacobip(n, a+1,b+1,τ) * (n+a+2)/(2n+a+b+4)
                @test jacobip(n+1,a,b+1,τ) ≈ jacobip(n+1,a+1,b+1,τ) * (n+a+b+3)/(2n+a+b+4) - jacobip(n,a+1,b+1,τ) * (n+b+2)/(2n+a+b+4)
                @test 2t*jacobip(n,a,b+2,τ) ≈ (τ+1) * jacobip(n,a,b+2,τ) ≈ jacobip(n+1,a,b+1,τ)*(2*(n+1)/(2n+a+b+3)) + jacobip(n,a,b+1,τ) * (2*(n+b+2)/(2n+a+b+3))
                @test t*jacobip(n-1,a+1,b+2,τ) ≈ (τ+1)/2 * jacobip(n-1,a+1,b+2,τ) ≈ jacobip(n,a+1,b+1,τ)*(n/(2n+a+b+2)) + jacobip(n-1,a+1,b+1,τ) * ((n+b+1)/(2n+a+b+2))

                @test τ*jacobip(n,a+1,b+1,τ) ≈ jacobip(n+1,a+1,b+1,τ)*(2*(n+1)*(n+a+b+3)/((2n+a+b+3)*(2n+a+b+4))) + jacobip(n,a+1,b+1,τ) * (b*(b + 2) - a*(a + 2))/((2n+a+b+2)*(2n+a+b+4)) + jacobip(n-1,a+1,b+1,τ) * (2*(n+a+1)*(n+b+1)/((2n+a+b+2)*(2n+a+b+3)))
                @test (1-t)*jacobip(n,a+1,b+1,τ) ≈ (1-τ)/2 * jacobip(n,a+1,b+1,τ) ≈ - jacobip(n+1,a+1,b+1,τ)*((n+1)*(n+a+b+3)/((2n+a+b+3)*(2n+a+b+4))) + jacobip(n,a+1,b+1,τ) * (a^2+2*(n+1)*(n+b+2) + a*(2n+b+4))/((2n+a+b+2)*(2n+a+b+4)) - jacobip(n-1,a+1,b+1,τ) * ((n+a+1)*(n+b+1)/((2n+a+b+2)*(2n+a+b+3)))

                @test (1-t)*jacobip(n,a+1,b,τ) ≈ (1-τ)/2 * jacobip(n,a+1,b,τ) ≈ - jacobip(n+1,a+1,b,τ)*((n+1)*(n+a+b+2)/((2n+a+b+2)*(2n+a+b+3))) + jacobip(n,a+1,b,τ) * (a^2+2*(n+1)*(n+b+1) + a*(2n+b+3))/((2n+a+b+1)*(2n+a+b+3)) - jacobip(n-1,a+1,b,τ) * ((n+a+1)*(n+b)/((2n+a+b+1)*(2n+a+b+2))) ≈ jacobip(n+1,a+1,b+1,τ) * (-(n+a+b+3)/(2n+a+b+4)*((n+1)*(n+a+b+2)/((2n+a+b+2)*(2n+a+b+3)))) + jacobip(n, a+1,b+1,τ) * (((2+a+b+n)*(a^2+a*(3+b+n)+(1+n)*(2+2b+n)))/((1+a+b+2n)*(2+a+b+2n)*(4+a+b+2n))) + jacobip(n-1,a+1,b+1,τ) * ((1+a+n)*(a^2-b*(1+b)+n+n^2+a*(2+n)))/((a+b+2n)*(2+a+b+2n)*(3+a+b+2n)) - jacobip(n-2,a+1,b+1,τ) * (n+a)*(n+a+1)*(n+b)/((2n+a+b)*(2n+a+b+1)*(2n+a+b+2))
            end


            𝐯 = (2n+a+b+3)/(2n+2a+b+3) * SVector(-2t*(n+a+1)*jacobip(n,a,b+2,τ) + (b+1)*(1-t)*jacobip(n,a+1,b+1,τ), (b+1)*jacobip(n,a+1,b+1,τ))

            @test 𝐯 ≈ -jacobip(n+1,a+1,b+1,τ) * ((1+n)*(3+a+b+n))/((4+a+b+2n)) * SVector(1,0) +
                    jacobip(n,a+1,b+1,τ) * (2n+a+b+3)/(2n+2a+b+3) * SVector(((n+b+2)/(2n+a+b+4) * (n+a+1)*(2*(n+1)/(2n+a+b+3)) - (n+a+b+2)/(2n+a+b+2) * (n+a+1)* (2*(n+b+2)/(2n+a+b+3)) + (b+1)*(a^2+2*(n+1)*(n+b+2) + a*(2n+b+4))/((2n+a+b+2)*(2n+a+b+4))),(b+1)) +
                    jacobip(n-1,a+1,b+1,τ) *((1+a+n)*(1+b+n)*(3+b+2n))/((2+a+b+2n) * (2n+2a+b+3)) * SVector(1,0)

            𝐪¹ = (2n+a+b+1)*(2n+2a+b+3) / ((n+a+1)*(2n+a+b+3) * 𝔯(n,a,b)) * ((2n+a+b+2)*(2n+a+b+3)*jacobip(n,a+1,b,τ) * SVector(1-t,1) - (n+a+b+2)*𝐯)

            @test vectorjacobip(n, 2, (a,a+1), b, t) ≈ 𝐪¹ ≈ (2n+a+b+1)*(2n+2a+b+3) / ((n+a+1)*(2n+a+b+3) * 𝔯(n,a,b)) * (
                                                        jacobip(n, a+1,b+1,τ) * (1+a+n)*(2+a+b+n)*(3+a+b+2n)/(3+2a+b+2n) * [((2(1+a+b)*(2+a+b)+(7+4a+5b)*n+2n^2))/((1+a+b+2n)*(2+a+b+2n)), 2] +
                                                        jacobip(n-1,a+1,b+1,τ) * (n+a+1) * [(2n+a+b+2)*(2n+a+b+3)* ((a^2-b*(1+b)+n+n^2+a*(2+n)))/((a+b+2n)*(2+a+b+2n)*(3+a+b+2n)) -  (n+a+b+2)*(1+b+n)*(3+b+2n)/((2+a+b+2n) * (2n+2a+b+3)), (2n+a+b+3) ] +
            - jacobip(n-2,a+1,b+1,τ) *(2n+a+b+3)* (n+a)*(n+a+1)*(n+b)/((2n+a+b)*(2n+a+b+1)) * [1,0]
                )


            @test vectorjacobip(n, 1, (a+1,a+1), b, t) ≈ t*jacobip(n-1, a+1, b+2, τ)*[1,-1] ≈ jacobip(n,a+1,b+1,τ)*(n/(2n+a+b+2)) * [1,-1] + jacobip(n-1,a+1,b+1,τ) * ((n+b+1)/(2n+a+b+2)) * [1,-1]
            @test vectorjacobip(n, 2, (a+1,a+1), b, t) ≈ jacobip(n, a+1, b, τ)*[1,1] ≈ jacobip(n,a+1,b+1,τ) * (n+a+b+2)/(2n+a+b+2) * [1,1] + jacobip(n-1,a+1,b+1,τ) * (n+a+1)/(2n+a+b+2) * [1,1]

            @test (2n+a+b+2)/n * vectorjacobip(n, 1, (a+1,a+1), b, t) + (2n+a+b+2)/(n+a+b+2) * vectorjacobip(n, 2, (a+1,a+1), b, t) ≈ jacobip(n,a+1,b+1,τ) * [2,0] + jacobip(n-1,a+1,b+1,τ) * ((n+b+1)/n * [1,-1] + (n+a+1)/(n+a+b+2) * [1,1])
            @test (2n+a+b+2)/n * vectorjacobip(n, 1, (a+1,a+1), b, t) - (2n+a+b+2)/(n+a+b+2) * vectorjacobip(n, 2, (a+1,a+1), b, t) ≈ jacobip(n,a+1,b+1,τ) * [0,-2] + jacobip(n-1,a+1,b+1,τ) * ((n+b+1)/n * [1,-1] - (n+a+1)/(n+a+b+2) * [1,1])

            @test (1+a+n)*(2+a+b+n)*(3+a+b+2n)/(3+2a+b+2n) * ((2(1+a+b)*(2+a+b)+(7+4a+5b)*n+2n^2))/((1+a+b+2n)*(2+a+b+2n)) * ((2n+a+b+2)/n * vectorjacobip(n, 1, (a+1,a+1), b, t) + (2n+a+b+2)/(n+a+b+2) * vectorjacobip(n, 2, (a+1,a+1), b, t))/2 ≈
            (1+a+n)*(2+a+b+n)*(3+a+b+2n)/(3+2a+b+2n) * ((2(1+a+b)*(2+a+b)+(7+4a+5b)*n+2n^2))/((1+a+b+2n)*(2+a+b+2n)) * (jacobip(n,a+1,b+1,τ) * [2,0] + jacobip(n-1,a+1,b+1,τ) * ((n+b+1)/n * [1,-1] + (n+a+1)/(n+a+b+2) * [1,1]))/2

            @test vectorjacobip(n-1, 1, (a+1,a+1), b, t) ≈ t*jacobip(n-2, a+1, b+2, τ)*[1,-1] ≈ jacobip(n-1,a+1,b+1,τ)*((n-1)/(2n+a+b)) * [1,-1] + jacobip(n-2,a+1,b+1,τ) * ((n+b)/(2n+a+b)) * [1,-1]
            @test vectorjacobip(n-1, 2, (a+1,a+1), b, t) ≈ jacobip(n-1, a+1, b, τ)*[1,1] ≈ jacobip(n-1,a+1,b+1,τ) * (n+a+b+1)/(2n+a+b) * [1,1] + jacobip(n-2,a+1,b+1,τ) * (n+a)/(2n+a+b) * [1,1]
            @test vectorjacobip(n-1, 1, (a+1,a+1), b, t) * ((2n+a+b)/(n+b)) + vectorjacobip(n-1, 2, (a+1,a+1), b, t) * (2n+a+b)/(n+a) ≈
                jacobip(n-1,a+1,b+1,τ)*((n-1)/(n+b) * [1,-1] + (n+a+b+1)/(n+a) * [1,1]) + jacobip(n-2,a+1,b+1,τ) * [2,0]


            𝐯̃ = (2(n-1)+a+b+3)/(2(n-1)+2a+b+3) * SVector(-2t*((n-1)+a+1)*jacobip((n-1),a,b+2,τ) + (b+1)*(1-t)*jacobip((n-1),a+1,b+1,τ), (b+1)*jacobip((n-1),a+1,b+1,τ))
            @test 𝐯̃ ≈ -jacobip(n,a+1,b+1,τ) * (n*(2+a+b+n))/((2+a+b+2n)) * SVector(1,0) +
                    jacobip(n-1,a+1,b+1,τ) * (2n+a+b+1)/(2n+2a+b+1) * SVector(((n+b+1)/(2n+a+b+2) * (n+a)*(2n/(2n+a+b+1)) - (n+a+b+1)/(2n+a+b) * (n+a)* (2*(n+b+1)/(2n+a+b+1)) + (b+1)*(a^2+2n*(n+b+1) + a*(2n+b+2))/((2n+a+b)*(2n+a+b+2))),(b+1)) +
                    jacobip(n-2,a+1,b+1,τ) *((a+n)*(b+n)*(1+b+2n))/((a+b+2n) * (2n+2a+b+1)) * SVector(1,0)

            @test 𝐯̃  ≈ -(n*(2+a+b+n))/(2n) * vectorjacobip(n, 1, (a+1,a+1), b, t) +
            -n/2 * vectorjacobip(n, 2, (a+1,a+1), b, t) +
            ((a+n)*(1+b+2n))/(2*(2n+2a+b+1)) * vectorjacobip(n-1, 1, (a+1,a+1), b, t) +
            ((b+n)*(1+b+2n))/(2 * (2n+2a+b+1)) * vectorjacobip(n-1, 2, (a+1,a+1), b, t)

            @test vectorjacobip(n, 1, (a,a+1), b, t) ≈ -(𝐯̃/n + 𝐪¹)

            for b = 0:3, a = -1:4, n = 0:10
                @test vectorjacobip(n, 0, (a,a+1), b, t) ≈ 1/2 * vectorjacobip(n, 0, (a+1,a+1), b, t) +
                (n ≤ 0 ? [0,0] :  -(n+a+b+2)*(6n+4a+3b+5)/(2𝔯(n,a,b)) * vectorjacobip(n, 1, (a+1,a+1), b, t) +
                (n ≤ 1 ? [0,0] : -(2n+2a+b+3)*(n+a)/(2𝔯(n,a,b))  * vectorjacobip(n-1, 1, (a+1,a+1), b, t)) +
                -(2n+2a+b+3)*(n+b)/(2𝔯(n,a,b)) * vectorjacobip(n-1, 0, (a+1,a+1), b, t))

                @test vectorjacobip(n, 0, (a+1,a), b, t) ≈ 1/2 * vectorjacobip(n, 0, (a+1,a+1), b, t) +
                (n ≤ 0 ? [0,0] :  (n+a+b+2)*(6n+4a+3b+5)/(2𝔯(n,a,b)) * vectorjacobip(n, 1, (a+1,a+1), b, t) +
                (n ≤ 1 ? [0,0] : (2n+2a+b+3)*(n+a)/(2𝔯(n,a,b))  * vectorjacobip(n-1, 1, (a+1,a+1), b, t)) +
                -(2n+2a+b+3)*(n+b)/(2𝔯(n,a,b)) * vectorjacobip(n-1, 0, (a+1,a+1), b, t))
            end
            for b = 0:3, a = -1:4, n = 1:10
                @test vectorjacobip(n, 1, (a,a+1), b, t) ≈
                (2*(2+a+b+n)*(1+a+b+2n)*(2+a+b+2n))/(n*𝔯(n,a,b)) * vectorjacobip(n, 1, (a+1,a+1), b, t) +
            (n ≤ 1 ? [0,0] : -(2(a+n)*(1+b+n)*(1+a+b+2n)*(2+a+b+2n))/(
        n*(1+2a+b+2n)*𝔯(n,a,b) )  * vectorjacobip(n-1, 1, (a+1,a+1), b, t)) +
                    -((2*(b+n)*(1+b+n)*(1+a+b+2n)*(2+a+b+2n))/(
        n*(1+2a+b+2n)*𝔯(n,a,b))) * vectorjacobip(n-1, 0, (a+1,a+1), b, t)
                @test vectorjacobip(n, 1, (a+1,a), b, t) ≈
                -(2*(2+a+b+n)*(1+a+b+2n)*(2+a+b+2n))/(n*𝔯(n,a,b)) * vectorjacobip(n, 1, (a+1,a+1), b, t) +
            (n ≤ 1 ? [0,0] : (2(a+n)*(1+b+n)*(1+a+b+2n)*(2+a+b+2n))/(
        n*(1+2a+b+2n)*𝔯(n,a,b) )  * vectorjacobip(n-1, 1, (a+1,a+1), b, t)) +
                    -((2*(b+n)*(1+b+n)*(1+a+b+2n)*(2+a+b+2n))/(
        n*(1+2a+b+2n)*𝔯(n,a,b))) * vectorjacobip(n-1, 0, (a+1,a+1), b, t)
            end
        end

    end

    @testset "mass" begin
        P = VectorJacobi((0,0), 0)
        Q = VectorJacobi((-1,0), 0)
        @test (P'P)[1:5,1:5] ≈ [sum(P[t,k]'P[t,j] for t in 0..1) for k=1:5, j=1:5]
        @test (Q'Q)[1:5,1:5] ≈ [sum(Q[t,k]'Q[t,j] for t in 0..1) for k=1:5, j=1:5]
    end

    @testset "diff ladder operators" begin
        for a = -1:5, b = 1:5
            P = jacobi(a,b,UnitInterval())
            t = 0.1
            @test [[2diff(P)[t,k]*t,0] + P[t,k]*b*[1,1] for k=1:5] ≈ laddergrad(P)[0.1,1:5]
        end

        for a = -1:5, b = 0:5
            𝐏 = VectorJacobi((a+1,a), b)
            curl𝐏 = laddercurl(𝐏)
            for k = 1:5
                @test curl𝐏[0.1,k] ≈ (b+1)/0.1 * only(diff(𝐏[0.1,k])) + 2diff(last.(𝐏[:,k]))[0.1] atol=1E-10
            end
        end

        for a = -1:5, b = 1:5
            P = jacobi(a,b,UnitInterval())
            @test laddercurl(laddergrad(P))[0.1,1] ≈ 0 atol=1E-10
        end

        # [sum((1-t)^(a+1) * t^(b+1) * jacobip(k, a+1, b+1, 2t-1) * ((b+1)/t * only(diff(vectorjacobip(n, 1, (a+1,a), b, t))) + 2ForwardDiff.derivative(t -> vectorjacobip(n, 1, (a+1,a), b, t)[2], t)) for t in 0..1) for k = 1:5]        
    end
end