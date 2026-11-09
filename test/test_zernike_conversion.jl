using DeRhamOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, BlockArrays, LinearAlgebra, Test

cart2pol(x, y) = (hypot(x, y), atan(y, x))
ym(m, x, y) = ((r, θ) = cart2pol(x, y); r^abs(m) * exp(im * m * θ))
ρ(θ) = @SMatrix[cos(θ) -sin(θ); sin(θ) cos(θ)]
Σ(x, y) = @SMatrix[x^2-y^2 2x*y; 2x*y y^2-x^2]
cA(m) = m == 0 ? @SVector[1, im * one(m)] : @SVector[1, im * sign(m)]
cB(m) = @SVector[1, -im * sign(m)]
S() = @SMatrix[1 0; 0 -1]

function 𝐲m(m, x, y)
    r, θ = cart2pol(x, y)
    r^(abs(m) - 1) * ρ(θ) * cA(m) * exp(im * m * θ)
end
function zmj(m, j, λ, x, y)
    r = hypot(x, y)
    jacobip(j, λ, abs(m), 2r^2 - 1) * ym(m, x, y)
end
function 𝐳mj(m, j, λ, k, x, y)
    r = hypot(x, y)
    if k == 1
        jacobip(j, λ, abs(m) - 1, 2r^2 - 1) * 𝐲m(m, x, y)
    else
        jacobip(j - 1, λ, abs(m) + 1, 2r^2 - 1) * Σ(x, y) * 𝐲m(m, x, y)
    end
end

@testset "eigenvectors" begin
    for m in -5:5
        iszero(m) && continue
        for θ in -π:π/10:π
            @test ρ(θ) * cA(m) ≈ exp(-im * sign(m) * θ) * cA(m)
            @test ρ(θ) * cB(m) ≈ exp(im * sign(m) * θ) * cB(m)
        end
    end
end

for x in -1:0.1:1, y in -1:0.1:1
    r, θ = cart2pol(x, y)
    @test Σ(x, y) ≈ r^2 * ρ(2θ) * S()
end

for m in -5:5
    iszero(m) && continue
    @test abs(m - sign(m)) == abs(m) - 1
    @test abs(m + sign(m)) == abs(m) + 1
end

@testset "conversion" begin
    for m in -5:5, j in (iszero(m) ? (1:4) : (0:4)), λ in (0.5, 1.0, 2.0)
        r = 0.9 * rand()
        θ = 2π * rand()
        x, y = r * cos(θ), r * sin(θ)
        if iszero(m)
            @test 𝐳mj(0, j, λ, 1, x, y) ≈ (j + λ) / j * zmj(-1, j - 1, λ, x, y) * @SVector[1, im]
            @test 𝐳mj(0, j, λ, 2, x, y) ≈ zmj(1, j - 1, λ, x, y) * @SVector[1, -im]
        else
            @test 𝐳mj(m, j, λ, 1, x, y) ≈ zmj(m - sign(m), j, λ, x, y) * cA(m)
            j == 0 && continue
            @test 𝐳mj(m, j, λ, 2, x, y) ≈ zmj(m + sign(m), j - 1, λ, x, y) * cB(m)
        end
    end
end

@testset "coupling matrix: function values" begin
    for λ in (0.0, 0.5, 1.0, 2.0)
        M, J = 3, 3
        fhat1 = Dict((m, j) => randn(ComplexF64) for m in -M:M for j in 0:J)
        fhat2 = Dict((m, j) => randn(ComplexF64) for m in -M:M for j in 1:J)
        fhat1[(0, 0)] = 0
        get1(m, j) = get(fhat1, (m, j), zero(ComplexF64))
        get2(m, j) = get(fhat2, (m, j), zero(ComplexF64))
        c1 = Dict{Tuple{Int,Int},ComplexF64}()
        c2 = Dict{Tuple{Int,Int},ComplexF64}()
        for μ in -(M + 1):(M+1), k in 0:J
            if μ ≥ 1
                c1[(μ, k)] = get1(μ + 1, k) + get2(μ - 1, k + 1)
                c2[(μ, k)] = im * get1(μ + 1, k) - im * get2(μ - 1, k + 1)
            elseif μ == 0
                c1[(0, k)] = get1(1, k) + get1(-1, k)
                c2[(0, k)] = im * get1(1, k) - im * get1(-1, k)
            elseif μ == -1
                c1[(-1, k)] = get1(-2, k) + (k + 1 + λ) / (k + 1) * get1(0, k + 1)
                c2[(-1, k)] = -im * get1(-2, k) + im * (k + 1 + λ) / (k + 1) * get1(0, k + 1)
            else
                c1[(μ, k)] = get1(μ - 1, k) + get2(μ + 1, k + 1)
                c2[(μ, k)] = -im * get1(μ - 1, k) + im * get2(μ + 1, k + 1)
            end
        end

        r = 0.4 + 0.4 * rand()
        θ_pt = 2π * rand()
        x, y = r * cos(θ_pt), r * sin(θ_pt)

        # Direct evaluation
        f_direct = zero(SVector{2,ComplexF64})
        for m in -M:M, j in (iszero(m) ? (1:J) : (0:J))
            f_direct = f_direct + fhat1[(m, j)] * 𝐳mj(m, j, λ, 1, x, y)
        end
        for m in -M:M, j in 1:J
            f_direct = f_direct + fhat2[(m, j)] * 𝐳mj(m, j, λ, 2, x, y)
        end

        # Reconstruction
        f1_sc = sum(c1[(μ, k)] * zmj(μ, k, λ, x, y) for μ in -(M + 1):(M+1) for k in 0:J)
        f2_sc = sum(c2[(μ, k)] * zmj(μ, k, λ, x, y) for μ in -(M + 1):(M+1) for k in 0:J)

        @test f_direct[1] ≈ f1_sc
        @test f_direct[2] ≈ f2_sc
    end
end

@testset "coupling matrix: inverse transform" begin
    for λ in (0.0, 0.5, 1.0, 2.0)
        M, J = 3, 3

        c1 = Dict((μ, k) => randn(ComplexF64) for μ in -M:M for k in 0:J)
        c2 = Dict((μ, k) => randn(ComplexF64) for μ in -M:M for k in 0:J)

        # Apply inverse coupling formulas to recover vector Zernike coefficients
        fhat1 = Dict{Tuple{Int,Int},ComplexF64}()
        fhat2 = Dict{Tuple{Int,Int},ComplexF64}()
        for μ in -M:M, k in 0:J
            a, b = c1[(μ, k)], c2[(μ, k)]
            if μ ≥ 1
                fhat1[(μ + 1, k)] = (a - im * b) / 2
                fhat2[(μ - 1, k + 1)] = (a + im * b) / 2
            elseif μ == 0
                fhat1[(1, k)] = (a - im * b) / 2
                fhat1[(-1, k)] = (a + im * b) / 2
            elseif μ == -1
                fhat1[(-2, k)] = (a + im * b) / 2
                fhat1[(0, k + 1)] = (k + 1) / (2 * (k + 1 + λ)) * (a - im * b)
            else
                fhat1[(μ - 1, k)] = (a + im * b) / 2
                fhat2[(μ + 1, k + 1)] = (a - im * b) / 2
            end
        end

        r = 0.4 + 0.4 * rand()
        θ_pt = 2π * rand()
        x, y = r * cos(θ_pt), r * sin(θ_pt)

        # Scalar Zernike evaluation
        f1_sc = sum(c1[(μ, k)] * zmj(μ, k, λ, x, y) for μ in -M:M for k in 0:J)
        f2_sc = sum(c2[(μ, k)] * zmj(μ, k, λ, x, y) for μ in -M:M for k in 0:J)

        # Vector Zernike evaluation with recovered coefficients
        f_vec = zero(SVector{2,ComplexF64})
        for ((m, j), fv) in pairs(fhat1)
            f_vec = f_vec + fv * 𝐳mj(m, j, λ, 1, x, y)
        end
        for ((m, j), fv) in pairs(fhat2)
            f_vec = f_vec + fv * 𝐳mj(m, j, λ, 2, x, y)
        end

        @test f_vec[1] ≈ f1_sc
        @test f_vec[2] ≈ f2_sc
    end
end

@testset "coupling matrix: inversion" begin
    for λ in (0.0, 0.5, 1.0, 2.0), k in 0:4
        a, b = randn(ComplexF64), randn(ComplexF64)

        # μ ≥ 1 and μ = 0: C⁺ coupling
        c1 = a + b
        c2 = im * a - im * b
        @test (c1 - im * c2) / 2 ≈ a
        @test (c1 + im * c2) / 2 ≈ b

        # μ ≤ -2: C⁻ coupling
        c1 = a + b
        c2 = -im * a + im * b
        @test (c1 + im * c2) / 2 ≈ a
        @test (c1 - im * c2) / 2 ≈ b

        # μ = -1: C⁻ D_k coupling
        α = (k + 1 + λ) / (k + 1)
        c1 = a + α * b
        c2 = -im * a + im * α * b
        @test (c1 + im * c2) / 2 ≈ a
        @test (k + 1) / (2 * (k + 1 + λ)) * (c1 - im * c2) ≈ b
    end
end

