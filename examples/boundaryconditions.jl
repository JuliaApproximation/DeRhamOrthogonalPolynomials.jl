using DeRhamOrthogonalPolynomials, MultivariateOrthogonalPolynomials, StaticArrays, LinearAlgebra, Test
using ContinuumArrays: grid

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
