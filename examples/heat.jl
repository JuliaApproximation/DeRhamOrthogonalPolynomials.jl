using DeRhamExamples, DeRhamOrthogonalPolynomials, MultivariateOrthogonalPolynomials, ClassicalOrthogonalPolynomials, ContinuumArrays, StaticArrays
using MultivariateOrthogonalPolynomials: ComplexZernikeTransform, zernikegrid, ModalTrav
diskcolumn(n, j) = (-1)^iseven(j) * j ÷ 2, 1:(n - (j+2) ÷ 4)

###############
# Heat equation
#
# u_t =  ∇⋅A(𝐱)∇u
################



# Start with standard Heat. We will use a simple backward Euler time stepper.
# We first precompute
function plan_heat_disk(h, M, N)
    [((m,kr) = diskcolumn(M, j); # We need to consider subindices of F
        Qₘ = DiskBasis(m, -1); # H¹ basis
        Pₘ = DiskBasis(m); # L² basis
        Δ = Symmetric((grad(Qₘ)'grad(Qₘ))[kr,kr]); # weak Laplacian
        Mass = Symmetric((Qₘ'Qₘ)[kr,kr]);
        (cholesky(Mass + h*Δ), Mass)) for j in 1:N]
end

function heatstep_disk!(Pl, U₀)
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
@time Pl = plan_heat_disk(h, size(U₀)...);
@time U = [ComplexZernikeTransform(N, -1) * u₀.(g)]

@time for _ = 1:1000
    push!(U, heatstep_disk!(Pl, last(U)))
end

using CairoMakie
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




function plan_heat_disk(v, h, M, N)
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
@time Pl = plan_heat_disk(v, h, size(U₀)...);
@time U = [ComplexZernikeTransform(N, -1) * u₀.(g)]

@time for _ = 1:1000
    push!(U, heatstep_disk!(Pl, last(U)))
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



######
# Now we do the above with a periodic cylinder beginning with standard Heat
######


# function plan_heat_cylinder(h, M, N)
#     [((m,kr) = diskcolumn(M, j); # We need to consider subindices of F
#         Qₘ = DiskBasis(m, -1); # H¹ basis
#         Pₘ = DiskBasis(m); # L² basis
#         Δ = Symmetric((grad(Qₘ)'grad(Qₘ))[kr,kr]); # weak Laplacian
#         Mass = Symmetric((Qₘ'Qₘ)[kr,kr]);
#         (cholesky(Mass + h*Δ), Mass)) for j in 1:N]
# end


u₀ = splat((x,y,z) -> exp(-5*(x-0.1)^2-10*(y-0.2)^2 - 20*(z-1.3)^2))
N,M = 20,300
g = vcat.(zernikegrid(N), reshape(grid(Fourier(), M),1,1,:))
U₀ = transform(Laurent(), complexzerniketransform(u₀.(g), (1,2)), 3)
𝐱 = (0.1,0.2,1.3)
# exp(im*k*z) does not depend on the disk mode, so we first sum over the Fourier modes
# and then evaluate the resulting disk expansion once
function cylindereval(U₀, (x,y,z))
    F = Laurent()[z, axes(U₀,3)] # exp(im*k*z), in the same order as the third dimension of U₀
    V = reshape(reshape(U₀, :, size(U₀,3)) * F, size(U₀,1), size(U₀,2))
    sum(begin
            (m,kr) = diskcolumn(size(V,1), j)
            transpose(DiskBasis(m)[SVector(x,y),kr]) * view(V,kr,j)
        end for j in axes(V,2))
end
@time cylindereval(U₀, 𝐱)
@test u₀(𝐱) ≈ cylindereval(U₀, 𝐱) atol=1E-3


# Start with standard Heat. We will use a simple backward Euler time stepper.
# We first precompute
function plan_heat_cylinder(h, M, N, L)
    [begin
        (m,kr) = diskcolumn(M, j) # We need to consider subindices of F
        k = (-1)^iseven(ℓ) * ℓ ÷ 2
        Qₘ = PeriodicCylinderBasis(m, k, -1) # H¹ basis
        Pₘ = PeriodicCylinderBasis(m, k) # L² basis
        Δ = Symmetric((grad(Qₘ)'grad(Qₘ))[kr,kr]) # weak Laplacian
        Mass = Symmetric((Qₘ'Qₘ)[kr,kr])
        (cholesky(Mass + h*Δ), Mass)
    end for j in 1:N, ℓ in 1:L]
end

function heatstep_cylinder!(Pl, U₀)
    U₁ = similar(U₀)
    for ℓ in axes(U₀,3), j in axes(U₀,2)
        m,kr = diskcolumn(size(U₀,1), j)
        L, M = Pl[j, ℓ]
        ldiv!(L, mul!(view(U₁, kr, j, ℓ), M, view(U₀,kr,j, ℓ)))
    end
    U₁
end

h = 0.01
@time U₀ = transform(Laurent(), complexzerniketransform(u₀.(g), (1,2)), 3);
@time Pl = plan_heat_cylinder(h, size(U₀)...);
U = [U₀];

@time for _ = 1:1000
    push!(U, heatstep_cylinder!(Pl, last(U)))
end

using GLMakie
GLMakie.activate!()


@time F = real(complexzernikeitransform(itransform(Laurent(), U[5], 3), (1,2)));


fig = Figure(;size=(1000,1000))
ax = Axis3(fig[1,1]; aspect=:equal, titlesize=60, titlegap=-120)
n = 20
X = repeat(range(-1, 1, n), 1, n)
Y = zeros(n,n)
Z = repeat(range(0, 2π,n)', n, 1)
R = range(0,1,n ÷ 2)
Θ = range(0,2π,n)'
surface!(ax, X, Y, Z; color=u₀.(SVector.(X, Y, Z)))
surface!(ax, Y, X, Z; color=u₀.(SVector.(Y, X, Z)))
let (X,Y,Z) = (R .* cos.(Θ), R .* sin.(Θ), fill(1.0π, n÷2, n))
    surface!(ax, X, Y, Z; color=u₀.(SVector.(X, Y, Z)))
end
fig
g[:,1,:]


    
    hidedecorations!(ax)
    hidespines!(ax)
    surface!(ax, X, Z, Y; color=f.(xs, 0, ys'))
    surface!(ax, Z, X, Y; color=f.(0, xs, ys'))