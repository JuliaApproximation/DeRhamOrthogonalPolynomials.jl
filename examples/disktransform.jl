using MultivariateOrthogonalPolynomials, DeRhamOrthogonalPolynomials, StaticArrays, LinearAlgebra, Test
using ContinuumArrays: grid
using ClassicalOrthogonalPolynomials: jacobip

M = 50 # total degree / modes
f₁ = [exp(x*cos(3*(y-1))) for (x,y) in UnitDisk()]
f₂ = [cos(x*sin(3*(y-1))) for (x,y) in UnitDisk()]
F₁  = (ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ f₁).matrix
F₂  = (ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ f₂).matrix

# column j holds the coefficients in DiskVector(m) for m = (-1)^iseven(j) * j ÷ 2.
G = disk2diskvector(F₁, F₂)
n = size(G,1)

𝐱 = SVector(0.1,0.2)
@test sum((m = (-1)^iseven(j) * j ÷ 2;
           sum(DiskVector(m)[𝐱,1:n] .* G[:,j])) for j = 1:2M+3) ≈ [f₁[𝐱], f₂[𝐱]]

# the same from values sampled on a grid
P = DiskVectorPlan(M+1)
𝐅 = [SVector(f₁[𝐲], f₂[𝐲]) for 𝐲 in grid(P)]
@test P * 𝐅 ≈ G
@test inv(P) * G ≈ 𝐅

# the matrix version, for an equivariant A(𝐱) = α(r²)*I + δ(r²)*ρ(π/2) + β(r²)*Σ(𝐱) + γ(r²)*ρ(π/2)*Σ(𝐱)
ρ = θ -> @SMatrix [cos(θ) -sin(θ); sin(θ) cos(θ)] # rotation by θ
Σ = 𝐱 -> @SMatrix [𝐱[1]^2-𝐱[2]^2 2𝐱[1]*𝐱[2]; 2𝐱[1]*𝐱[2] 𝐱[2]^2-𝐱[1]^2]
A = 𝐱 -> (t = 𝐱'𝐱; exp(t)*I + cos(3t)*ρ(π/2) + sin(2t)*Σ(𝐱) + ρ(π/2)*Σ(𝐱)/(2+t))
f₁₁ = [A(SVector(x,y))[1,1] for (x,y) in UnitDisk()]
f₁₂ = [A(SVector(x,y))[1,2] for (x,y) in UnitDisk()]
f₂₁ = [A(SVector(x,y))[2,1] for (x,y) in UnitDisk()]
f₂₂ = [A(SVector(x,y))[2,2] for (x,y) in UnitDisk()]
F₁₁ = (ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ f₁₁).matrix
F₁₂ = (ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ f₁₂).matrix
F₂₁ = (ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ f₂₁).matrix
F₂₂ = (ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ f₂₂).matrix

# g[4k-3:4k] holds the k-th coefficients of α, δ in Jacobi(0,0) and of β, γ in Jacobi(0,2), in r²
g = disk2diskmatrix(F₁₁, F₁₂, F₂₁, F₂₂)
t, n = 𝐱'𝐱, length(g) ÷ 4
α, δ, β, γ = (sum(g[4k-4+ν] * jacobip(k-1, 0, b, 2t-1) for k = 1:n) for (ν,b) in ((1,0), (2,0), (3,2), (4,2)))
@test α*I + δ*ρ(π/2) + β*Σ(𝐱) + γ*ρ(π/2)*Σ(𝐱) ≈ A(𝐱)

# the same from values sampled on a grid
P = DiskMatrixPlan(M+1)
𝐀 = [A(SVector(𝐲[1],𝐲[2])) for 𝐲 in grid(P)]
@test P * 𝐀 ≈ g
@test inv(P) * g ≈ 𝐀
