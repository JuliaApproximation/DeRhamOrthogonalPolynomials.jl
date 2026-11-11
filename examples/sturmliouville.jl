using DeRhamOrthogonalPolynomials, MultivariateOrthogonalPolynomials, StaticArrays, LinearAlgebra, ForwardDiff, DeRhamExamples, Test
using ClassicalOrthogonalPolynomials: grammatrix
using ContinuumArrays: grid

########################
# Sturm–Liouville
#
#   -∇ ⋅ A∇u = f,   A(𝐱) = α(r²)*I + β(r²)*Σ(𝐱) + γ(r²)*ρ(π/2)*Σ(𝐱)
#
# where A is symmetric positive definite, can be written in weak form as
#
#   <∇v, A∇u> = <v, f>.
# 
# Multiplying by A keeps the mode of a vector field, so the problem splits
# into one small system per mode m. The computed solutions are compared with exact solutions u, for which
# f = -∇ ⋅ A∇u is computed by automatic differentiation.
########################

ρ = θ -> @SMatrix [cos(θ) -sin(θ); sin(θ) cos(θ)] # rotation by θ
Σ = 𝐱 -> @SMatrix [𝐱[1]^2-𝐱[2]^2 2𝐱[1]*𝐱[2]; 2𝐱[1]*𝐱[2] 𝐱[2]^2-𝐱[1]^2]
A = 𝐱 -> (t = sum(abs2,𝐱); (2+exp(t))*I + sin(2t)/2*Σ(𝐱) + ρ(π/2)*Σ(𝐱)/(2+t))
symdiskplot((x,y) -> A((x,y)))  # visualise the matrix field

Pₐ = DiskMatrixPlan(41) # exact up to degree 40
g = Pₐ * A.(grid(Pₐ)) # coefficients of α, δ, β, γ

# L(u, 𝐱) = -∇ ⋅ A∇u at 𝐱, which gives f for the exact solutions u below
div = (𝐯, 𝐱) -> tr(ForwardDiff.jacobian(𝐯, 𝐱))
L = (u, 𝐱) -> -div(𝐲 -> A(𝐲)*ForwardDiff.gradient(real∘u, 𝐲), 𝐱) - im*div(𝐲 -> A(𝐲)*ForwardDiff.gradient(imag∘u, 𝐲), 𝐱)
∇ = (u, 𝐱) -> ForwardDiff.gradient(real∘u, 𝐱) + im*ForwardDiff.gradient(imag∘u, 𝐱)

# test points
𝐱s = (SVector(0.1,0.2), SVector(-0.5,0.3), SVector(0.0,0.0), SVector(0.3,-0.6), SVector(0.7,0.7), SVector(cos(1.0),sin(1.0)))


# Zero Dirichlet condition: in mode m we use W = Weighted(DiskBasis(m, 1)), which vanishes on the boundary.

u = 𝐱 -> (1-sum(abs2,𝐱))*exp(𝐱[1])*cos(2𝐱[2]+1) # the exact solution

M = 30 # total degree / modes
F = (ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ [L(u, SVector(x,y)) for (x,y) in UnitDisk()]).matrix # coefficients of f in DiskBasis(m)
n = size(F,1)

uₙ = zeros(ComplexF64, length(𝐱s)) # the computed solution at the test points
for j = 1:2M+1
    m = (-1)^iseven(j) * j ÷ 2
    W, P, V = Weighted(DiskBasis(m, 1)), DiskBasis(m), DiskVector(m)
    D = V \ grad(W)                            # ∇W = V * D
    Aₘ = equivariantmultiplication(V, g)       # A * V = V * Aₘ
    S = (D' * grammatrix(V) * Aₘ * D)[1:n,1:n] # stiffness matrix <∇W, A∇W>
    R = (W'P)[1:n,1:n]                         # load matrix, so that <W, f> = R * F[:,j]
    c = S \ (R * F[:,j])                       # coefficients in W
    uₙ .+= [sum(W[𝐱,1:n] .* c) for 𝐱 in 𝐱s]
end
@test uₙ ≈ [u(𝐱) for 𝐱 in 𝐱s] atol=1e-13


# Natural boundary condition 𝐱'A∇u = 0: in mode m we use Q = DiskBasis(m, -1), as for Poisson with zero Neumann
# conditions. The constant is in the kernel, so it is left out and solutions are compared up to a constant.
# On the boundary 𝐱'A∇u = (α+β)*∂u/∂r + (γ-δ)*∂u/∂θ, which vanishes for each term of the exact solution.

ξ = (1 + im*(1/3 - cos(3)/2)/(2 + exp(1) + sin(2)/2))/2 # (1 + im*(γ-δ)/(α+β))/2 at r = 1
u = 𝐱 -> (t = sum(abs2,𝐱); cos(π*t) + (1-t)^2*exp(𝐱[1])*cos(2𝐱[2]+1) + (𝐱[1]+im*𝐱[2])*(1 + ξ*(1-t))) # the exact solution
for θ in (0.0, 1.0, 2.5, 4.0)
    𝐲 = SVector(cos(θ), sin(θ))
    @test abs(𝐲' * A(𝐲) * ∇(u, 𝐲)) ≤ 1e-13 # u satisfies the boundary condition
end

M = 36
F = (ComplexZernike()[:,Block.(Base.OneTo(M+1))] \ [L(u, SVector(x,y)) for (x,y) in UnitDisk()]).matrix
n = size(F,1)

uₙ = zeros(ComplexF64, length(𝐱s))
for j = 1:2M+1
    m = (-1)^iseven(j) * j ÷ 2
    Q, P, V = DiskBasis(m, -1), DiskBasis(m), DiskVector(m)
    kr = (1+iszero(m)):n # without the constant
    D = V \ grad(Q)
    Aₘ = equivariantmultiplication(V, g)
    S = (D' * grammatrix(V) * Aₘ * D)[kr,kr]
    E = (Q'P)[kr,1:n]
    c = S \ (E * F[:,j])
    uₙ .+= [sum(Q[𝐱,kr] .* c) for 𝐱 in 𝐱s]
end
e = uₙ - [u(𝐱) for 𝐱 in 𝐱s]
@test e ≈ fill(e[1], length(e)) atol=1e-13 # equal up to a constant
