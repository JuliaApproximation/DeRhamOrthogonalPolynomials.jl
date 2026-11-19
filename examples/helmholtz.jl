using DeRhamOrthogonalPolynomials, MultivariateOrthogonalPolynomials, StaticArrays, CairoMakie
using MultivariateOrthogonalPolynomials: ComplexZernikeTransform, zernikegrid

diskcolumn(F, j) = (-1)^iseven(j) * j ÷ 2, F[1:(size(F,1) - (j+2) ÷ 4), j]


##################
# (Inhomogeneous) Helmholtz Equation
#
#   (∇ ⋅ A(𝐱) ∇ + κ(𝐱)) u = f(𝐱)
#
# where κ is invariant and A is SPD nearly everywhere and equivariant to rotations, i.e., 
#
# A(𝐱) = a(r) * 𝐞_r*𝐞_r' + b(r) * 𝐞_θ*𝐞_θ' + c(r)*(𝐞_r*𝐞_θ' + 𝐞_θ*𝐞_r')
#
# f need not be equivariant.  In weak form the equation becomes find u ∈ H¹ such that for al
# v ∈ H¹ we have
#
#   ⟨∇v, A ∇u⟩ +  ⟨∇v, κ ∇u⟩ = ⟨v, f⟩
#
# We will focus on natural (Neumann) conditions.
#
# We first start with the standard Helmholtz (A = I and κ = ω^2).
##################

𝐱₀ = SVector(0.1,0.2)
f = splat((x,y) -> exp(-100x^2-150y^2) + cos(100x * y^2))

contourf([f(𝐱) for 𝐱 in UnitDisk()]; levels=20) # we have an off-centred Gaussian

# in this package we do the transforms manually (for now), to lay the groundwork
# future MPI support. 

ω = 32.5
N = 10ceil(Int,ω) # polynomials degree + 1, 7 * wave length
g = zernikegrid(N) # grid
F = ComplexZernikeTransform(N) * f.(g)
U = zero(F) # need zeros in extra entries

for j in axes(F,2)
    m,𝐟  = diskcolumn(F, j) # We need to consider subindices of F 
    Qₘ = DiskBasis(m, -1) # H¹ basis
    Pₘ = DiskBasis(m) # L² basis
    kr = axes(𝐟,1)
    Δ = -(grad(Qₘ)'grad(Qₘ))[kr,kr] # weak Laplacian
    M = (Qₘ'Qₘ)[kr,kr]
    R = (Qₘ'Pₘ)[kr,kr]
    U[kr,j] = (Δ + ω^2*M) \ (R*𝐟)
end

contourf(first.(g), last.(g), real(ComplexZernikeITransform(N) * U);levels=40)

