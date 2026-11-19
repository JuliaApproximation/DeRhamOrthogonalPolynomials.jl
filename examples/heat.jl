using DeRhamExamples, DeRhamOrthogonalPolynomials, CairoMakie
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
        Δ = (grad(Qₘ)'grad(Qₘ))[kr,kr]; # weak Laplacian
        Mass = (Qₘ'Qₘ)[kr,kr];
        R = (Qₘ'Pₘ)[kr,kr];
        (lu(Mass + h*Δ), Mass)) for j in 1:N]
end

function heatstep!(Pl, U₀)
    for j in axes(U₀,2)
        m,kr = diskcolumn(size(U₀,1), j)
        ldiv!(Pl[j][1], lmul!(Pl[j][2], view(U₀,kr,j)))
    end
    U₀
end


u₀ = splat((x,y) -> exp(-10*(x-0.1)^2-20*(y-0.2)^2))

contourf([u₀(𝐱) for 𝐱 in UnitDisk()])

N = 100
g = zernikegrid(N) # grid
U₀ = ComplexZernikeTransform(N) * u₀.(g)


h = 0.1
@time Pl = plan_heat(h, size(U₀)...);
@time heatstep!(Pl, U₀);

# Zernike()[ModalTrav(U₀)