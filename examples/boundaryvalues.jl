using DeRhamOrthogonalPolynomials, StaticArrays, LinearAlgebra, Test
using ContinuumArrays: grid

# The boundary values of a vector field can be read off its coefficients in the bases with a = -1.
M = 40 # total degree
# column j of the coefficients holds the mode m = (-1)^iseven(j) * j ÷ 2

𝐟 = 𝐱 -> SVector(cos(4𝐱[1] + 𝐱[2]), exp(𝐱[1]) * sin(𝐱[1] - 3𝐱[2]))
f_r = θ -> (𝐲=SVector(cos(θ), sin(θ)); dot(𝐲, 𝐟(𝐲)))               # normal component on the boundary
f_θ = θ -> (𝐲=SVector(cos(θ), sin(θ)); 𝐲[1]*𝐟(𝐲)[2] - 𝐲[2]*𝐟(𝐲)[1]) # tangential component on the boundary
θ = range(0, 2π; length=97)

# Only the first two elements of DiskVector(m, (-1,-1)) are nonzero on the boundary. For m ≠ 0 they are exp(im*m*θ)
# times (1, im*sign(m)) and (1, -im*sign(m)) in polar components, and for m = 0 they are 𝐞_r and 𝐞_θ.
P = DiskVectorPlan(M+1, -1)
G = P * 𝐟.(grid(P))
u_r = θ -> sum((m=(-1)^iseven(j) * j ÷ 2; (iszero(m) ? G[1, j] : G[1, j] + G[2, j]) * cis(m*θ)) for j in axes(G, 2))
u_θ = θ -> sum((m=(-1)^iseven(j) * j ÷ 2; (iszero(m) ? G[2, j] : im*sign(m)*(G[1, j] - G[2, j])) * cis(m*θ)) for j in axes(G, 2))
@test u_r.(θ) ≈ f_r.(θ)
@test u_θ.(θ) ≈ f_θ.(θ)

# so a field that vanishes on the boundary has no coefficients in the first two rows
@test norm((P*[(1-𝐱[1]^2-𝐱[2]^2)*𝐟(𝐱) for 𝐱 in grid(P)])[1:2, :]) ≤ 1e-13

# Only the first element of DiskVector(m, (-1,0)) has a normal component on the boundary, and only the first
# (the second for m = 0) of DiskVector(m, (0,-1)) has a tangential one.
G_n = DiskVectorPlan(M+1, (-1, 0)) * 𝐟.(grid(P))
G_t = DiskVectorPlan(M+1, (0, -1)) * 𝐟.(grid(P))
# the normal and tangential components on the boundary, each from a single row
n_r = θ -> sum((m=(-1)^iseven(j) * j ÷ 2; (iszero(m) ? G_n[1, j] : G_n[1, j]/2) * cis(m*θ)) for j in axes(G_n, 2))
t_θ = θ -> sum((m=(-1)^iseven(j) * j ÷ 2; (iszero(m) ? G_t[2, j] : im*sign(m)*G_t[1, j]/2) * cis(m*θ)) for j in axes(G_t, 2))
@test n_r.(θ) ≈ f_r.(θ)
@test t_θ.(θ) ≈ f_θ.(θ)
