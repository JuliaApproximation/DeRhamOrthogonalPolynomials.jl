using DeRhamOrthogonalPolynomials, MultivariateOrthogonalPolynomials, StaticArrays, InfiniteArrays, LazyBandedMatrices,  LinearAlgebra, Test, ForwardDiff
using ClassicalOrthogonalPolynomials: weightedgrammatrix, expand, jacobip
using DeRhamOrthogonalPolynomials: equivariantjacobimatrix, DiskCurlFree, 𝔯

isapproxdiagonal(A) = A ≈ Diagonal(diag(A))
gradient(f, 𝐱) = ForwardDiff.gradient(𝐲 -> real(f(𝐲)), 𝐱) + im*ForwardDiff.gradient(𝐲 -> imag(f(𝐲)), 𝐱)

function fd_curl(F, 𝐱) # g_x - f_y
    J_r = ForwardDiff.jacobian(𝐲 -> real(F(𝐲)), 𝐱)
    J_i = ForwardDiff.jacobian(𝐲 -> imag(F(𝐲)), 𝐱)
    J_r[2,1] - J_r[1,2] + im*(J_i[2,1] - J_i[1,2])
end
function fd_div(F, 𝐱) # f_x + g_y
    J_r = ForwardDiff.jacobian(𝐲 -> real(F(𝐲)), 𝐱)
    J_i = ForwardDiff.jacobian(𝐲 -> imag(F(𝐲)), 𝐱)
    J_r[1,1] + J_r[2,2] + im*(J_i[1,1] + J_i[2,2])
end


include("test_harmonicpolynomials.jl")
include("test_vectorjacobi.jl")
include("test_disk.jl")
include("test_ball.jl")
include("test_pdes.jl")

# include("test_periodiccylinder.jl")
