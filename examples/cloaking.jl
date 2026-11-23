using DeRhamOrthogonalPolynomials, MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays, LinearAlgebra, SpecialFunctions, ForwardDiff, CairoMakie, Test
using MultivariateOrthogonalPolynomials: ComplexZernikeTransform, ComplexZernikeITransform, zernikegrid
using FastGaussQuadrature: gausslegendre

########################
# Cloaking
# 
#   (Δ + V₂(r²))u = f in r < ϱ,   (Δ + V₁(r²))u = 0 in ϱ < r < 1,   u = 0 on r = 1,
#
# that is, (Δ + V)u = f with V = V₂ in r < ϱ and V = V₁ in r > ϱ, which in weak form is
#
#   -<∇v, ∇u> + <v, Vu> = <v, f>   for all v with v = 0 on r = 1.
#
# The problem splits into one system per mode m since the potential V is symmetric. In mode m we expand u in W = Weighted(DiskBasis(m, 1)),
# which vanishes on r = 1, and f in P = DiskBasis(m), and compute <W, VW> by quadrature. As V jumps at r = ϱ, we
# replace it by a smooth Vξ that equals V outside the layer ϱ - ξ/2 < r < ϱ + ξ/2, and split the quadrature at the
# edges of the layer.
########################

diskcolumn(F, j) = (-1)^iseven(j) * j ÷ 2, F[1:(size(F, 1)-(j+2)÷4), j] # the mode of column j of F, and its entries
diskindex(m) = m ≥ 0 ? 2m+1 : -2m # the column of mode m

ψ = s -> s > 0 ? exp(-1/s) : 0.0
Φ = s -> ψ(s) / (ψ(s) + ψ(1 - s)) # smooth, 0 for s ≤ 0 and 1 for s ≥ 1, with Φ(s) + Φ(1-s) = 1
window = (V₂, V₁, ϱ, ξ) -> t -> (c=iszero(ξ) ? float(t > ϱ^2) : Φ((sqrt(t) - ϱ)/ξ + 1/2); (1 - c)*V₂(t) + c*V₁(t)) # Vξ in t = r², V₀ the step
layer = (ϱ, ξ) -> iszero(ξ) ? (0, ϱ^2, 1) : (0, (ϱ - ξ/2)^2, (ϱ + ξ/2)^2, 1) # Vξ is V₂, the window and V₁ between these values of t

# Φ, and Vξ for V₂ = 144, V₁ = 16 and ϱ = 1/2 as in the layered disk below, with ξ = 0 the step
fig = Figure(size=(1000, 400))
s = -0.5:0.001:1.5
ax = Axis(fig[1, 1]; xlabel="s", title="Φ(s) = ψ(s) / (ψ(s) + ψ(1 - s))")
lines!(ax, s, ψ.(s); label="ψ(s) = exp(-1/s)")
lines!(ax, s, ψ.(1 .- s); label="ψ(1 - s)")
lines!(ax, s, Φ.(s); label="Φ(s)")
axislegend(ax; position=:lt)
r = 0.3:0.0002:0.7
ax = Axis(fig[1, 2]; xlabel="r", title="Vξ")
vspan!(ax, 0.4, 0.6; color=(:gray, 0.1)) # the layer for ξ = 0.2
for ξ in (0.2, 0.1, 0.05, 0.0)
    lines!(ax, r, window(t -> 144, t -> 16, 0.5, ξ).(r .^ 2); label="ξ = $ξ", linestyle=iszero(ξ) ? :dash : :solid)
end
axislegend(ax)
fig

# π ∫ V(t) tᵖ P_k^(0,|μ|)(2t-1) P_j^(0,|ν|)(2t-1) dt for 0 ≤ k, j < n, by Gauss–Legendre quadrature on each [ts[i], ts[i+1]]
function radialgram(μ, ν, p, V, ts, n)
    x, w = gausslegendre(n + p ÷ 2 + 50)
    sum(zip(ts[1:(end-1)], ts[2:end])) do (a, b)
        t = a .+ (b - a) .* (x .+ 1) ./ 2
        Pμ, Pν = Jacobi{Float64}(0, abs(μ))[2t .- 1, 1:n], Jacobi{Float64}(0, abs(ν))[2t .- 1, 1:n]
        π * (b - a)/2 * Pμ' * ((w .* V.(t) .* t .^ p) .* Pν)
    end
end

# the coefficients of u from those of f (from ComplexZernikeTransform) for V smooth on each [ts[i], ts[i+1]]
function cloak(F, V, ts=(0, 1))
    U = zero(F)
    for j in axes(F, 2)
        m, 𝐟 = diskcolumn(F, j)
        iszero(𝐟) && continue
        n = length(𝐟) - 1
        W, P = Weighted(DiskBasis(m, 1)), DiskBasis(m)
        L = (P\W)[1:(n+1), 1:n]
        A = L' * radialgram(m, m, abs(m), V, ts, n+1) * L - (grad(W)'grad(W))[1:n, 1:n] # <W, VW> - <∇W, ∇W>
        b = (W'P)[1:n, 1:(n+1)] * 𝐟 # <W, f>
        U[1:(n+1), j] = L * (A \ b)
    end
    U
end

Δ = (w, 𝐱) -> tr(ForwardDiff.hessian(real∘w, 𝐱)) + im*tr(ForwardDiff.hessian(imag∘w, 𝐱)) # Laplacian of complex w

# A manufactured solution, for a smooth V

V = t -> 20 + 10tanh(4(t - 0.3))
u = 𝐱 -> (t=sum(abs2, 𝐱); (1 - t) * (exp(𝐱[1] + im*𝐱[2]/2) * cos(3t) + (𝐱[1] - im*𝐱[2])^3)) # u = 0 on r = 1
f = 𝐱 -> Δ(u, 𝐱) + V(sum(abs2, 𝐱)) * u(𝐱)

N = 61
g = zernikegrid(N)
U = cloak(ComplexZernikeTransform(N) * f.(g), V)
@test ComplexZernikeITransform(N) * U ≈ u.(g)

# Layered disk: for V₂ = s₂², V₁ = s₁² constant, f = 0 and g = Σ g_m exp(im*m*θ), mode m of the exact solution is
# a J_m(s₂r) in r < ρ and b J_m(s₁r) + c Y_m(s₁r) in ρ < r < 1. As u'' jumps at r = ρ, the error decreases
# algebraically with the number of coefficients n.

s₁, s₂, ϱ = 4.0, 12.0, 0.5
M = 20 # modes |m| ≤ M
fₘ = m -> 1/(1 + m^2)
function layeredmode(m)
    J, Y = z -> besselj(m, z), z -> bessely(m, z)
    dJ, dY = z -> (besselj(m-1, z) - besselj(m+1, z))/2, z -> (bessely(m-1, z) - bessely(m+1, z))/2
    e = 1/s₁^2 - 1/s₂^2
    a, b, c = [J(s₂*ϱ) -J(s₁*ϱ) -Y(s₁*ϱ);
        s₂*dJ(s₂*ϱ) -s₁*dJ(s₁*ϱ) -s₁*dY(s₁*ϱ);
        0 J(s₁) Y(s₁)] \ [ϱ^abs(m)*e, abs(m)*ϱ^(abs(m)-1)*e, -1/s₁^2]
    r -> r < ϱ ? a*J(s₂*r) + r^abs(m)/s₂^2 : b*J(s₁*r) + c*Y(s₁*r) + r^abs(m)/s₁^2
end
uₘ = Dict(m => layeredmode(m) for m in (-M):M)
u = 𝐱 -> sum(fₘ(m) * uₘ[m](norm(𝐱)) * cis(m*atan(𝐱[2], 𝐱[1])) for m in (-M):M)

N = 201 # at most 100 unknowns per mode
g = zernikegrid(N)
F = zeros(ComplexF64, size(g))
F[1, diskindex.((-M):M)] = fₘ.((-M):M) # as DiskBasis(m)[:,1] = r^|m| exp(im*m*θ)
uₑ = u.(g)

errs = map((0.02, 0.01, 0.002, 0.0)) do ξ
    U = cloak(F, window(t -> s₂^2, t -> s₁^2, ϱ, ξ), layer(ϱ, ξ))
    norm(ComplexZernikeITransform(N) * U - uₑ, Inf) / norm(uₑ, Inf)
end
@test errs[1]/errs[2] ≈ 4 rtol=0.1 # quadratic error in ξ
@test errs[3] ≤ 1.2/100^2 # ξ = 0.2/n
@test errs[4] ≤ 1.2/100^2 # the step


# A Gaussian source in the hole, with N = 201 and ξ = 0.2/100 as above

ξ = 0.002
f = 𝐱 -> exp(-((𝐱[1]-0.1)^2 + (𝐱[2]-0.1)^2)/0.08^2)
U = cloak(ComplexZernikeTransform(N) * f.(g), window(t -> s₂^2, t -> s₁^2, ϱ, ξ), layer(ϱ, ξ))
fig, ax, _ = tricontourf(vec(first.(g)), vec(last.(g)), vec(real(ComplexZernikeITransform(N) * U)); levels=40,
    axis=(; aspect=DataAspect(), title="Re u"))
lines!(ax, ϱ*cos.(0:0.01:2π), ϱ*sin.(0:0.01:2π); color=:white) # the interface r = ϱ
fig

########################
# Vector version
#
#   (Δ + V)𝐮 = 𝐟,   𝐮 = 0 on r = 1,   V = α(r²)*I + δ(r²)*ρ(π/2) + β(r²)*Σ(𝐱) + γ(r²)*ρ(π/2)*Σ(𝐱),
#
# with Δ acting on each component. In the components w₋ = u₁ - im*u₂ and w₊ = u₁ + im*u₂ this reads
#
#   Δw₋ + (α - im*δ)w₋ + (β - im*γ)(x - im*y)²w₊ = f₋,   Δw₊ + (α + im*δ)w₊ + (β + im*γ)(x + im*y)²w₋ = f₊,
#
# so if w₋ has mode m - 1 then w₊ has mode m + 1, and the two are coupled only through β and γ. Each is expanded in
# Weighted(DiskBasis(m ± 1, 1)) as in the scalar case, and u₁ = (w₋ + w₊)/2, u₂ = im*(w₋ - w₊)/2.
########################

# the coefficients of w₋ and w₊ from those of f₋ and f₊
function vectorcloak(F₋, F₊, (α, δ, β, γ), ts=(0, 1))
    U₋, U₊ = zero(F₋), zero(F₊)
    M = size(F₋, 2) ÷ 2 # scalar modes |μ| ≤ M
    for m in (-(M-1)):(M-1)
        μ, ν = m-1, m+1 # the modes of w₋ and w₊
        𝐟₋, 𝐟₊ = last(diskcolumn(F₋, diskindex(μ))), last(diskcolumn(F₊, diskindex(ν)))
        n = min(length(𝐟₋), length(𝐟₊)) - 1
        W₋, P₋, W₊, P₊ = Weighted(DiskBasis(μ, 1)), DiskBasis(μ), Weighted(DiskBasis(ν, 1)), DiskBasis(ν)
        L₋, L₊ = (P₋\W₋)[1:(n+1), 1:n], (P₊\W₊)[1:(n+1), 1:n]
        p = (abs(μ) + abs(ν)) ÷ 2 + 1 # r^|μ| r^|ν| |x ± im*y|² = tᵖ
        A₋₋ = L₋' * radialgram(μ, μ, abs(μ), t -> α(t) - im*δ(t), ts, n+1) * L₋ - (grad(W₋)'grad(W₋))[1:n, 1:n]
        A₋₊ = L₋' * radialgram(μ, ν, p, t -> β(t) - im*γ(t), ts, n+1) * L₊
        A₊₋ = L₊' * radialgram(ν, μ, p, t -> β(t) + im*γ(t), ts, n+1) * L₋
        A₊₊ = L₊' * radialgram(ν, ν, abs(ν), t -> α(t) + im*δ(t), ts, n+1) * L₊ - (grad(W₊)'grad(W₊))[1:n, 1:n]
        c = [A₋₋ A₋₊; A₊₋ A₊₊] \ [(W₋'P₋)[1:n, 1:(n+1)] * 𝐟₋[1:(n+1)]; (W₊'P₊)[1:n, 1:(n+1)] * 𝐟₊[1:(n+1)]]
        U₋[1:(n+1), diskindex(μ)], U₊[1:(n+1), diskindex(ν)] = L₋ * c[1:n], L₊ * c[(n+1):end]
    end
    U₋, U₊
end


# A manufactured solution, with α, δ, β, γ nonzero

ρ = θ -> @SMatrix [cos(θ) -sin(θ); sin(θ) cos(θ)] # rotation by θ
Σ = 𝐱 -> @SMatrix [𝐱[1]^2-𝐱[2]^2 2𝐱[1]*𝐱[2]; 2𝐱[1]*𝐱[2] 𝐱[2]^2-𝐱[1]^2]
α, δ, β, γ = t -> 20 + 5t, t -> 3cos(2t), t -> 2 + sin(3t), t -> 1/(2 + t)
V = 𝐱 -> (t=sum(abs2, 𝐱); α(t)*I + δ(t)*ρ(π/2) + β(t)*Σ(𝐱) + γ(t)*ρ(π/2)*Σ(𝐱))
𝐮 = 𝐱 -> (1 - sum(abs2, 𝐱)) * SVector(exp(𝐱[1])*cos(𝐱[2]) + im*𝐱[2]^2, sin(𝐱[1] + 2𝐱[2]) - im*exp(𝐱[2])*𝐱[1]) # 𝐮 = 0 on r = 1
𝐟 = 𝐱 -> SVector(Δ(𝐲 -> 𝐮(𝐲)[1], 𝐱), Δ(𝐲 -> 𝐮(𝐲)[2], 𝐱)) + V(𝐱) * 𝐮(𝐱)

N = 61
g = zernikegrid(N)
T, Tⁱ = ComplexZernikeTransform(N), ComplexZernikeITransform(N)
𝐅 = 𝐟.(g)
U₋, U₊ = vectorcloak(T * (first.(𝐅) - im*last.(𝐅)), T * (first.(𝐅) + im*last.(𝐅)), (α, δ, β, γ))
w₋, w₊ = Tⁱ * U₋, Tⁱ * U₊
@test SVector.((w₋ + w₊)/2, im*(w₋ - w₊)/2) ≈ 𝐮.(g)


# A vector source in the hole, with V = s₂² I in r < ϱ and V = a𝐞_r𝐞_r' + b𝐞_θ𝐞_θ' in ϱ < r < 1, so that there
# α = (a+b)/2 and β = (a-b)/(2r²), and ξ = 0.2/50 for the at most 50 unknowns per mode

a, b = 9.0, 25.0
V₂ = (t -> s₂^2, t -> 0.0, t -> 0.0, t -> 0.0)
V₁ = (t -> (a+b)/2, t -> 0.0, t -> (a-b)/(2t), t -> 0.0)
ξ = 0.004
𝐟 = 𝐱 -> exp(-((𝐱[1]-0.1)^2 + (𝐱[2]-0.1)^2)/0.08^2) * SVector(1, 0)

N = 101
g = zernikegrid(N)
T, Tⁱ = ComplexZernikeTransform(N), ComplexZernikeITransform(N)
𝐅 = 𝐟.(g)
U₋, U₊ = vectorcloak(T * (first.(𝐅) - im*last.(𝐅)), T * (first.(𝐅) + im*last.(𝐅)), window.(V₂, V₁, ϱ, ξ), layer(ϱ, ξ))
w₋, w₊ = Tⁱ * U₋, Tⁱ * U₊
fig, ax, _ = tricontourf(vec(first.(g)), vec(last.(g)), vec(norm.(SVector.((w₋ + w₊)/2, im*(w₋ - w₊)/2))); levels=40,
    axis=(; aspect=DataAspect(), title="|𝐮|"))
lines!(ax, ϱ*cos.(0:0.01:2π), ϱ*sin.(0:0.01:2π); color=:white)
fig
