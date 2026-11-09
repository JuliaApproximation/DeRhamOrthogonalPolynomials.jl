using DeRhamOrthogonalPolynomials, MultivariateOrthogonalPolynomials, StaticArrays, LinearAlgebra, Test
using ContinuumArrays: grid, coefficients, plan_transform

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
