using DeRhamOrthogonalPolynomials, HarmonicOrthogonalPolynomials, MultivariateOrthogonalPolynomials, ForwardDiff, LinearAlgebra, DomainSets, StaticArrays, Test
using DeRhamOrthogonalPolynomials: harmonicpolynomial, vectorharmonicpolynomial,  vectorharmonicpolynomials, vectorharmonicpolynomialr, vectorharmonicpolynomialx, sphericalharmonicy_θ, vectorharmonicpolynomial
using ForwardDiff: derivative
using HarmonicOrthogonalPolynomials: SphericalCoordinate

laplacian(f, 𝐱) = tr(ForwardDiff.hessian(𝐲 -> real(f(𝐲)), 𝐱)) + im*tr(ForwardDiff.hessian(𝐲 -> imag(f(𝐲)), 𝐱))

@testset "Harmonic Polynomials" begin
    @testset "2D" begin
        𝐱 = SVector(0.1,0.2)
        for m = -5:5
            @test abs(laplacian(𝐱 -> harmonicpolynomial(m, 𝐱), 𝐱)) ≤ 1E-13
        end

        @test isapproxdiagonal([sum(conj(harmonicpolynomial(m, 𝐱))harmonicpolynomial(ℓ, 𝐱) for 𝐱 in UnitDisk()) for m=-3:3, ℓ=-3:3])

        for m = setdiff(-5:5, 0)
            @test gradient(𝐱 -> harmonicpolynomial(m, 𝐱), 𝐱) ≈ abs(m) * vectorharmonicpolynomial(m, 𝐱)
        end
    end

    @testset "3D" begin
        𝐱 = SVector(0.1,0.2,0.3)
        for ℓ = 0:5, m = -ℓ:ℓ
            @test abs(laplacian(𝐱 -> harmonicpolynomial(ℓ, m, 𝐱), 𝐱)) ≤ 1E-13
        end

        @test sum(sum(r^2 for r in 0.0..1) for 𝐱 in UnitSphere()) ≈ 4π/3
        for ℓ = 0:3, m = -ℓ:ℓ, n = 0:3, o = -n:n
            if ℓ ≠ n || m ≠ o
                @test abs(sum(sum(conj(harmonicpolynomial(ℓ, m, r*𝐱))harmonicpolynomial(n, o, r*𝐱) * r^2 for r in 0.0..1) for 𝐱 in UnitSphere())) ≤ 1E-10
            end
        end

        @testset "partial" begin
            x,y,z = 𝐱
            (; r,θ,φ) = SphericalCoordinate(𝐱)

            for ℓ = 0:3, m = -ℓ:ℓ
                # if m > 0
                #     @test sqrt(1-x^2) * derivative(x -> associatedlegendre(ℓ, m, x), x) ≈ ((ℓ+m)*(ℓ-m+1)*associatedlegendre(ℓ, m-1, x) - associatedlegendre(ℓ,m+1,x))/2
                #     @test (1-x^2) * derivative(x -> associatedlegendre(ℓ, m, x), x) ≈ (ℓ+1)*x*associatedlegendre(ℓ,m,x) - (ℓ-m+1)*associatedlegendre(ℓ+1,m,x)
                # end

                @test gradient(𝐱 -> harmonicpolynomial(ℓ, m, 𝐱), 𝐱) ≈ vectorharmonicpolynomials(ℓ, m, 𝐱)

                if ℓ ≠ 0
                    @test derivative(θ -> associatedlegendre(ℓ, m, cos(θ)), θ) ≈ m*cot(θ)*associatedlegendre(ℓ,m,cos(θ)) + (m < ℓ) *  associatedlegendre(ℓ,m+1,cos(θ))
                    @test derivative(θ -> sphericalharmonicy(ℓ, m, θ, φ), θ) ≈ sphericalharmonicy_θ(ℓ, m, θ, φ)
                    @test derivative(φ -> sphericalharmonicy(ℓ, m, θ, φ), φ) ≈ im*m*sphericalharmonicy(ℓ, m, θ, φ)
                else
                    @test  gradient(𝐱 -> harmonicpolynomial(ℓ, m, 𝐱), 𝐱) == [0,0,0]
                end

                @test 𝐱 × vectorharmonicpolynomials(ℓ, m, 𝐱) ≈ vectorharmonicpolynomialx(ℓ, m, 𝐱)
                @test vectorharmonicpolynomialr(ℓ, m, 𝐱) ≈ 𝐱 * harmonicpolynomial(ℓ, m, 𝐱)
            end
        end
    end
end