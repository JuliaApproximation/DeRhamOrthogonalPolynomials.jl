using DeRhamOrthogonalPolynomials, DomainSets, Test
using DeRhamOrthogonalPolynomials: zernike, vectorzernike, zernikegrad

@testset "Ball" begin
    𝐱 = SVector(0.1,0.2,0.3)
    @testset "BallBasis" begin
        for ℓ = 0:3, m = -ℓ:ℓ
            @test isapproxdiagonal([sum(sum(conj(BallBasis(ℓ, m)[r*𝐱,k])BallBasis(ℓ, m)[r*𝐱,j] * r^2 for r in 0.0..1) for 𝐱 in UnitSphere()) for k=1:4, j=1:4])
        end
    end

    @testset "BallVector" begin
        for ℓ = 0:3, m = -ℓ:ℓ
            @test isapproxdiagonal([sum(sum(vectorzernike(+, ℓ, m, k, r*𝐱)'vectorzernike(+, ℓ, m, j, r*𝐱) for 𝐱 in UnitSphere()) * r^2 for r in 0.0..1) for k=0:3, j=0:3]) 
            @test isapproxdiagonal([sum(sum(vectorzernike(-, ℓ, m, k, r*𝐱)'vectorzernike(-, ℓ, m, j, r*𝐱) for 𝐱 in UnitSphere()) * r^2 for r in 0.0..1) for k=0:3, j=0:3]) 
            @test isapproxdiagonal([sum(sum(vectorzernike(×, ℓ, m, k, r*𝐱)'vectorzernike(×, ℓ, m, j, r*𝐱) for 𝐱 in UnitSphere()) * r^2 for r in 0.0..1) for k=0:3, j=0:3]) 
        end

        @test isapproxdiagonal([sum(sum(vectorzernike(+, ℓ, 0, 0, r*𝐱)'vectorzernike(+, ℓ2, 0, 0, r*𝐱) for 𝐱 in UnitSphere()) * r^2 for r in 0.0..1) for ℓ=0:3, ℓ2=0:3])
        @test isapproxdiagonal([sum(sum(vectorzernike(+, 2, m, 0, r*𝐱)'vectorzernike(+, 2, n, 0, r*𝐱) for 𝐱 in UnitSphere()) * r^2 for r in 0.0..1) for m=-2:2, n=-2:2])
        @test norm([sum(sum(vectorzernike(+, ℓ, 0, 0, r*𝐱)'vectorzernike(-, ℓ2, 0, 0, r*𝐱) for 𝐱 in UnitSphere()) * r^2 for r in 0.0..1) for ℓ=0:3, ℓ2=1:3]) ≤ 1E-13
        @test norm([sum(sum(vectorzernike(+, ℓ, 0, 0, r*𝐱)'vectorzernike(×, ℓ2, 0, 0, r*𝐱) for 𝐱 in UnitSphere()) * r^2 for r in 0.0..1) for ℓ=0:3, ℓ2=1:3]) ≤ 1E-13
        @test norm([sum(sum(vectorzernike(+, ℓ, 0, 0, r*𝐱)'vectorzernike(-, ℓ2, 0, 0, r*𝐱) for 𝐱 in UnitSphere()) * r^2 for r in 0.0..1) for ℓ=1:3, ℓ2=1:3]) ≤ 1E-13
    end

    @testset "BallNormal" begin
        for ℓ = 0:3, m = -ℓ:ℓ, j = 0:3
            @test gradient(𝐱 -> zernike(ℓ, m, j, -1, 𝐱), 𝐱) ≈ zernikegrad(ℓ, m, j, -1, 𝐱)
        end
    end
end