using DeRhamExamples, DeRhamOrthogonalPolynomials, MultivariateOrthogonalPolynomials, StaticArrays, CairoMakie
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


h = 0.01
@time Pl = plan_heat(h, size(U₀)...);
@time U = [ComplexZernikeTransform(N, -1) * u₀.(g)]

@time for _ = 1:1000
    push!(U, heatstep!(Pl, last(U)))
end


K = 100
t = Observable(1)
jr = axes(ModalTrav(U[1]),1)
𝐮 = @lift( real(ComplexZernike(-1)[:,jr] * ModalTrav(U[$t])))
    
fig, ax, plt = contourf(𝐮; levels=range(0,1,101), colormap=:balance, extendlow=:auto, extendhigh=:auto)
Colorbar(fig[1,2], plt)

record(fig, "heat.mp4", 1:K+1; framerate=5) do i
    t[] = i
    ax.title[] = "t = $(round((i-1)*h, digits=2))"
end


# Now add A




function plan_heat(v, h, M, N)
    [((m,kr) = diskcolumn(M, j); # We need to consider subindices of F 
        Qₘ = DiskBasis(m, -1); # H¹ basis
        Pₘ = DiskBasis(m); # L² basis
        Aₘ = equivariantmultiplication(DiskVector(m), v);
        Vₘ = DiskVector(m);
        Δ = Hermitian(((Vₘ\grad(Qₘ))'Aₘ * (Vₘ\grad(Qₘ)))[kr,kr]); # weak Laplacian
        Mass = Symmetric((Qₘ'Qₘ)[kr,kr]);
        (lu(Mass + h*Δ), Mass)) for j in 1:N]
end
A = splat((x,y) -> SVector(x,y) * SVector(x,y)')
A = splat((x,y) -> SVector(-y,x) * SVector(-y,x)')
A = splat((x,y) -> I + SVector(x,y) * SVector(-y,x)' + SVector(-y,x) * SVector(x,y)')
v = DiskMatrixPlan(3) * A.(zernikegrid(3))


h = 0.01
@time Pl = plan_heat(v, h, size(U₀)...);
@time U = [ComplexZernikeTransform(N, -1) * u₀.(g)]

@time for _ = 1:1000
    push!(U, heatstep!(Pl, last(U)))
end


K = 20
t = Observable(1)
jr = axes(ModalTrav(U[1]),1)
𝐮 = @lift( real(ComplexZernike(-1)[:,jr] * ModalTrav(U[$t])))
    
fig, ax, plt = contourf(𝐮; levels=range(0,1,101), colormap=:balance, extendlow=:auto, extendhigh=:auto)
Colorbar(fig[1,2], plt)

record(fig, "heat_radial.mp4", 1:K+1; framerate=5) do i
    t[] = i
    ax.title[] = "t = $(round((i-1)*h, digits=2))"
end
