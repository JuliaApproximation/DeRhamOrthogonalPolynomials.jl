using DeRhamExamples, DeRhamOrthogonalPolynomials, MultivariateOrthogonalPolynomials, CairoMakie
using MultivariateOrthogonalPolynomials: ComplexZernikeTransform, zernikegrid, ModalTrav

###############
# Heat equation
#
# u_t =  ∇⋅A(𝐱)∇u
################


diskcolumn(n, j) = (-1)^iseven(j) * j ÷ 2, 1:(n - (j+2) ÷ 4)

# Start with standard Heat. We will use a simple backward Euler time stepper. 
# We first precompute 
function plan_heat(h, M, N)
    [((m,kr) = diskcolumn(M, j); # We need to consider subindices of F 
        Qₘ = DiskBasis(m, -1); # H¹ basis
        Pₘ = DiskBasis(m); # L² basis
        Δ = Symmetric((grad(Qₘ)'grad(Qₘ))[kr,kr]); # weak Laplacian
        Mass = Symmetric((Qₘ'Qₘ)[kr,kr]);
        (cholesky(Mass + h*Δ), Mass)) for j in 1:N]
end

function heatstep!(Pl, U₀)
    U₁ = similar(U₀)
    for j in axes(U₀,2)
        m,kr = diskcolumn(size(U₀,1), j)
        L, M = Pl[j]
        ldiv!(L, mul!(view(U₁, kr, j), M, view(U₀,kr,j)))
    end
    U₁
end


u₀ = splat((x,y) -> exp(-10*(x-0.1)^2-20*(y-0.2)^2))


N = 100
g = zernikegrid(N) # grid
U₀ = ComplexZernikeTransform(N, -1) * u₀.(g)

contourf(ComplexZernike(-1)[:,axes(ModalTrav(U₀),1)] * ModalTrav(U₀))

h = 0.1
@time Pl = plan_heat(h, size(U₀)...);
U = [U₀]

@time for _ = 1:100
    push!(U, heatstep!(Pl, last(U)))
end


ComplexZernike(-1)