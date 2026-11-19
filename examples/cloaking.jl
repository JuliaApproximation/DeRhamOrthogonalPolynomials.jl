using DeRhamOrthogonalPolynomials, MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, LinearAlgebra, SpecialFunctions, ForwardDiff, CairoMakie, Test
using MultivariateOrthogonalPolynomials: ComplexZernikeTransform, ComplexZernikeITransform, zernikegrid
using FastGaussQuadrature: gaussjacobi, gausslegendre

#
# Cloaking
#
#   (Δ + V₁(r))u = 0 in ρ < r < 1,   (Δ + V₂(r))u = f in r < ρ,   ∂u/∂r = g on r = 1.
#
#  We write this as (Δ + V)u = f in the unit disk, with V = V₂ for r < ρ and V₁ for r > ρ.
#

# π ∫₀¹ W tᵖ P_k^(0,|μ|) P_j^(0,|ν|) dt, W = W₂ for t < ρ², W₁ for t > ρ²
function radialgram(μ, ν, p, W₂, W₁, ρ, n; nq=n + p ÷ 2 + 20)
    Pμ, Pν = Jacobi{Float64}(0, abs(μ)), Jacobi{Float64}(0, abs(ν))
    x, w = gaussjacobi(nq, 0.0, Float64(p)) # t in [0, ρ²], with the weight t^p
    t = ρ^2 .* (x .+ 1) ./ 2
    I₂ = ρ^(2p+2) / 2.0^(p+1) * Pμ[2t .- 1, 1:n]' * ((w .* W₂.(t)) .* Pν[2t .- 1, 1:n])
    x, w = gausslegendre(nq)                # t in [ρ², 1]
    t = ρ^2 .+ (1 - ρ^2) .* (x .+ 1) ./ 2
    I₁ = (1 - ρ^2)/2 * Pμ[2t .- 1, 1:n]' * ((w .* W₁.(t) .* t .^ p) .* Pν[2t .- 1, 1:n])
    π * (I₂ + I₁)
end

# mode m of u in DiskBasis(m, -1)[:,1:n]
function cloakmode(m, n, (V₂, V₁, ρ), 𝐟, g)
    Q, P = DiskBasis(m, -1), DiskBasis(m)
    R = (P\Q)[1:n, 1:n] # Q = P * R, upper bidiagonal
    A = R' * radialgram(m, m, abs(m), V₂, V₁, ρ, n) * R - (grad(Q)'grad(Q))[1:n, 1:n] # weak form
    b = complex((Q'P)[1:n, 1:length(𝐟)] * 𝐟)
    b[1] -= 2π * g
    A \ b
end

𝐱s = vec([SVector(r*cos(θ), r*sin(θ)) for r in (0.0, 0.2, 0.45, 0.55, 0.8, 1.0), θ in (0.3, 2.0, 4.0)])

V = t -> 20 + 10tanh(4(t - 0.3))
u = 𝐱 -> exp(𝐱[1] + im*𝐱[2]/2) * cos(3sum(abs2, 𝐱)) + (𝐱[1] - im*𝐱[2])^3
Δ = (w, 𝐱) -> tr(ForwardDiff.hessian(w, 𝐱))
f = 𝐱 -> Δ(real∘u, 𝐱) + im*Δ(imag∘u, 𝐱) + V(sum(abs2, 𝐱)) * u(𝐱)
∂ᵣu = 𝐱 -> 𝐱' * (ForwardDiff.gradient(real∘u, 𝐱) + im*ForwardDiff.gradient(imag∘u, 𝐱))
fourier = (w, m) -> sum(w(SVector(cos(θ), sin(θ))) * cis(-m*θ) for θ in 2π*(0:255)/256)/256 # mode m on r = 1

N = 61
F = ComplexZernikeTransform(N) * f.(zernikegrid(N)) # coefficients of f in DiskBasis(m), column j for mode m
uₙ = zeros(ComplexF64, length(𝐱s))
for j in axes(F, 2)
    m, nⱼ = (-1)^iseven(j) * j ÷ 2, size(F, 1) - (j+2) ÷ 4 # the mode and |column j|
    c = cloakmode(m, nⱼ, (V, V, 0.5), F[1:nⱼ, j], fourier(∂ᵣu, m))
    uₙ .+= [sum(DiskBasis(m, -1)[𝐱, 1:nⱼ] .* c) for 𝐱 in 𝐱s]
end
@test uₙ ≈ u.(𝐱s)


# Layered disk: for V₂ = s₂², V₁ = s₁² constant, f = 0 and g = Σ g_m exp(im*m*θ), mode m of the exact solution is
# a J_m(s₂r) in r < ρ and b J_m(s₁r) + c Y_m(s₁r) in ρ < r < 1. As u'' jumps at r = ρ, the error decreases
# algebraically with the number of coefficients n.
function layeredexact(m, s₁, s₂, ρ, g) # (a, b, c), from the continuity of u and ∂u/∂r at r = ρ and ∂u/∂r = g at r = 1
    J, Y = z -> besselj(m, z), z -> bessely(m, z)
    dJ, dY = z -> (besselj(m-1, z) - besselj(m+1, z))/2, z -> (bessely(m-1, z) - bessely(m+1, z))/2
    [J(s₂*ρ) -J(s₁*ρ) -Y(s₁*ρ);
        s₂*dJ(s₂*ρ) -s₁*dJ(s₁*ρ) -s₁*dY(s₁*ρ);
        0 s₁*dJ(s₁) s₁*dY(s₁)] \ [0, 0, g]
end

s₁, s₂, ρ = 4.0, 12.0, 0.5
M = 20 # modes |m| ≤ M
gₘ = m -> 1/(1 + m^2)
uₑ = map(𝐱s) do 𝐱
    r, θ = norm(𝐱), atan(𝐱[2], 𝐱[1])
    sum(((a, b, c)=layeredexact(m, s₁, s₂, ρ, gₘ(m)); (r < ρ ? a*besselj(m, s₂*r) : b*besselj(m, s₁*r) + c*bessely(m, s₁*r)) * cis(m*θ)) for m in (-M):M)
end
errs = map((100, 200)) do n
    uₙ = zeros(ComplexF64, length(𝐱s))
    for m in (-M):M
        c = cloakmode(m, n, (t -> s₂^2, t -> s₁^2, ρ), Float64[], gₘ(m))
        uₙ .+= [sum(DiskBasis(m, -1)[𝐱, 1:n] .* c) for 𝐱 in 𝐱s]
    end
    maximum(abs, uₙ - uₑ) / maximum(abs, uₑ)
end
@test errs[1] ≤ 1e-3
@test errs[2] ≤ 1e-4


# A source in the hole and g = 0: f is a Gaussian centred in r < ρ. The values on the grid of the inverse
# transform follow after converting each mode from DiskBasis(m, -1) to DiskBasis(m).

f = 𝐱 -> exp(-((𝐱[1]-0.1)^2 + (𝐱[2]-0.1)^2)/0.08^2)
N = 201
F = ComplexZernikeTransform(N) * f.(zernikegrid(N))
U = zero(F)
for j in axes(F, 2)
    m, nⱼ = (-1)^iseven(j) * j ÷ 2, size(F, 1) - (j+2) ÷ 4
    U[1:nⱼ, j] = (DiskBasis(m)\DiskBasis(m, -1))[1:nⱼ, 1:nⱼ] * cloakmode(m, nⱼ, (t -> s₂^2, t -> s₁^2, ρ), F[1:nⱼ, j], 0)
end
vals = ComplexZernikeITransform(N) * U # the solution on zernikegrid(N)
g = zernikegrid(N)
fig = Figure()
ax = Axis(fig[1,1]; aspect=DataAspect(), title="Re u")
contourf!(ax, first.(g), last.(g), real(vals); levels=40)
lines!(ax, ρ*cos.(0:0.01:2π), ρ*sin.(0:0.01:2π); color=:white) # the interface r = ρ
fig

#
# Vector version
#
#   (Δ + V(𝐱))𝐮 = 𝐟,   V = α(r²)*I + δ(r²)*ρ(π/2) + β(r²)*Σ(𝐱) + γ(r²)*ρ(π/2)*Σ(𝐱),   ∂𝐮/∂r = 𝐠 on r = 1,
#
# with Δ acting on each component and (α, δ, β, γ) different for r < ρ and r > ρ. In the components
# w₋ = u₁ - im*u₂ and w₊ = u₁ + im*u₂, that is, 𝐲₋₁'𝐮 and 𝐲₁'𝐮, Δ acts on each, and
#
#   (V𝐮)₋ = (α - im*δ) w₋ + (β - im*γ) r² exp(-2im*θ) w₊,   (V𝐮)₊ = (α + im*δ) w₊ + (β + im*γ) r² exp(2im*θ) w₋,
#
# so in vector mode m, w₋ has scalar mode m-1 and w₊ has scalar mode m+1, coupled only through β and γ. Each is
# expanded in DiskBasis(m ± 1, -1) as in the scalar case, and 𝐮 = (w₋*𝐲₁ + w₊*𝐲₋₁)/2.
#

# vector mode m of w₋ in DiskBasis(m-1, -1)[:,1:n] and of w₊ in DiskBasis(m+1, -1)[:,1:n], for Vᵢ = (α, δ, β, γ)
function vectorcloakmode(m, n, (V₂, V₁, ρ), 𝐟₋, 𝐟₊, g₋, g₊)
    μ, ν = m-1, m+1
    Q₋, P₋, Q₊, P₊ = DiskBasis(μ, -1), DiskBasis(μ), DiskBasis(ν, -1), DiskBasis(ν)
    R₋, R₊ = (P₋\Q₋)[1:n, 1:n], (P₊\Q₊)[1:n, 1:n]
    W = (V, s) -> t -> V[1](t) + s*im*V[2](t) # α ± im*δ
    C = (V, s) -> t -> V[3](t) + s*im*V[4](t) # β ± im*γ
    p = (abs(μ) + abs(ν)) ÷ 2 + 1 # the power of t from r² and the factors r^|μ|, r^|ν|
    A = [R₋' * radialgram(μ, μ, abs(μ), W(V₂, -1), W(V₁, -1), ρ, n) * R₋ - (grad(Q₋)'grad(Q₋))[1:n, 1:n] R₋' * radialgram(μ, ν, p, C(V₂, -1), C(V₁, -1), ρ, n) * R₊;
        R₊' * radialgram(ν, μ, p, C(V₂, 1), C(V₁, 1), ρ, n) * R₋ R₊' * radialgram(ν, ν, abs(ν), W(V₂, 1), W(V₁, 1), ρ, n) * R₊ - (grad(Q₊)'grad(Q₊))[1:n, 1:n]]
    b = complex([(Q₋'P₋)[1:n, 1:length(𝐟₋)] * 𝐟₋; (Q₊'P₊)[1:n, 1:length(𝐟₊)] * 𝐟₊])
    b[1] -= 2π * g₋
    b[n+1] -= 2π * g₊
    c = A \ b
    c[1:n], c[(n+1):2n]
end

column = μ -> μ > 0 ? 2μ+1 : max(1, -2μ) # the column of mode μ


# a manufactured soln with α, δ, β, γ nonzero.

ρπ₂ = @SMatrix [0 -1; 1 0] # ρ(π/2)
Σ = 𝐱 -> @SMatrix [𝐱[1]^2-𝐱[2]^2 2𝐱[1]*𝐱[2]; 2𝐱[1]*𝐱[2] 𝐱[2]^2-𝐱[1]^2]
α, δ, β, γ = t -> 20 + 5t, t -> 3cos(2t), t -> 2 + sin(3t), t -> 1/(2 + t)
Vₘ = 𝐱 -> (t=sum(abs2, 𝐱); α(t)*I + δ(t)*ρπ₂ + β(t)*Σ(𝐱) + γ(t)*ρπ₂*Σ(𝐱))
𝐮 = 𝐱 -> SVector(exp(𝐱[1])*cos(𝐱[2]) + im*𝐱[2]^2, sin(𝐱[1] + 2𝐱[2]) - im*exp(𝐱[2])*𝐱[1])
w₋, w₊ = 𝐱 -> 𝐮(𝐱)[1] - im*𝐮(𝐱)[2], 𝐱 -> 𝐮(𝐱)[1] + im*𝐮(𝐱)[2]
Δw = (w, 𝐱) -> Δ(real∘w, 𝐱) + im*Δ(imag∘w, 𝐱)
𝐟 = 𝐱 -> SVector(Δw(𝐲 -> 𝐮(𝐲)[1], 𝐱), Δw(𝐲 -> 𝐮(𝐲)[2], 𝐱)) + Vₘ(𝐱) * 𝐮(𝐱)
∂ᵣ = w -> 𝐱 -> 𝐱' * (ForwardDiff.gradient(real∘w, 𝐱) + im*ForwardDiff.gradient(imag∘w, 𝐱))

N = 61
𝐅 = 𝐟.(zernikegrid(N))
F₋, F₊ = ComplexZernikeTransform(N) * (first.(𝐅) - im*last.(𝐅)), ComplexZernikeTransform(N) * (first.(𝐅) + im*last.(𝐅))
nₛ = μ -> size(F₋, 1) - (column(μ)+2) ÷ 4 # the number of coefficients of scalar mode μ
𝐮ₙ = zeros(SVector{2,ComplexF64}, length(𝐱s))
for m in (-(N-2)):(N-2)
    μ, ν = m-1, m+1
    n = min(nₛ(μ), nₛ(ν))
    c₋, c₊ = vectorcloakmode(m, n, ((α, δ, β, γ), (α, δ, β, γ), 0.5), F₋[1:n, column(μ)], F₊[1:n, column(ν)],
        fourier(∂ᵣ(w₋), μ), fourier(∂ᵣ(w₊), ν))
    for (i, 𝐱) in enumerate(𝐱s)
        a₋, a₊ = sum(DiskBasis(μ, -1)[𝐱, 1:n] .* c₋), sum(DiskBasis(ν, -1)[𝐱, 1:n] .* c₊)
        𝐮ₙ[i] += SVector((a₋ + a₊)/2, im*(a₋ - a₊)/2)
    end
end
@test 𝐮ₙ ≈ 𝐮.(𝐱s)


# A vector source in the hole with 𝐠 = 0

a, b = 9.0, 25.0
V₂ = (t -> s₂^2, t -> 0.0, t -> 0.0, t -> 0.0)
V₁ = (t -> (a+b)/2, t -> 0.0, t -> (a-b)/(2t), t -> 0.0)
𝐟 = 𝐱 -> exp(-((𝐱[1]-0.1)^2 + (𝐱[2]-0.1)^2)/0.08^2) * SVector(1, 0)
N = 101
𝐅 = 𝐟.(zernikegrid(N))
F₋, F₊ = ComplexZernikeTransform(N) * (first.(𝐅) - im*last.(𝐅)), ComplexZernikeTransform(N) * (first.(𝐅) + im*last.(𝐅))
nₛ = μ -> size(F₋, 1) - (column(μ)+2) ÷ 4
W₋, W₊ = zero(F₋), zero(F₊) # coefficients of w₋ and w₊ in DiskBasis(μ)
for m in (-(N-2)):(N-2)
    μ, ν = m-1, m+1
    n = min(nₛ(μ), nₛ(ν))
    c₋, c₊ = vectorcloakmode(m, n, (V₂, V₁, ρ), F₋[1:n, column(μ)], F₊[1:n, column(ν)], 0, 0)
    W₋[1:n, column(μ)] = (DiskBasis(μ)\DiskBasis(μ, -1))[1:n, 1:n] * c₋
    W₊[1:n, column(ν)] = (DiskBasis(ν)\DiskBasis(ν, -1))[1:n, 1:n] * c₊
end
Sⁱ = ComplexZernikeITransform(N)
vals₁, vals₂ = Sⁱ * ((W₋ + W₊)/2), Sⁱ * (im*(W₋ - W₊)/2) # the components of 𝐮 on zernikegrid(N)
g = zernikegrid(N)
fig = Figure()
ax = Axis(fig[1,1]; aspect=DataAspect(), title="|𝐮|")
contourf!(ax, first.(g), last.(g), sqrt.(abs2.(vals₁) .+ abs2.(vals₂)); levels=40)
lines!(ax, ρ*cos.(0:0.01:2π), ρ*sin.(0:0.01:2π); color=:white)
fig
