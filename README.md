# DeRhamOrthogonalPolynomials.jl
A Julia package for orthogonal polynomials that discretise de Rham complexes

[![CI](https://github.com/JuliaApproximation/DeRhamOrthogonalPolynomials.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/JuliaApproximation/DeRhamOrthogonalPolynomials.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JuliaApproximation/DeRhamOrthogonalPolynomials.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaApproximation/DeRhamOrthogonalPolynomials.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This experimental package builds scalar and vector-valued orthogonal polynomial bases on simple geometries
so that `grad` and `curl` map between bases with sparse operators. It builds on
[ClassicalOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/ClassicalOrthogonalPolynomials.jl),
[MultivariateOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/MultivariateOrthogonalPolynomials.jl) and
[HarmonicOrthogonalPolynomials.jl](https://github.com/JuliaApproximation/HarmonicOrthogonalPolynomials.jl).

Currently supported:

- **Disk**: Fourier mode-by-mode scalar bases (`DiskBasis`), vector bases (`DiskVector`) and the
  curl-free/co-curl-free decomposition (`DiskCurlFree`, `DiskCoCurlFree`), with `grad`, `curl` and `rotgrad`.
- **Ball**: `BallBasis` and vector Zernike polynomials (`vectorzernike`).
- **Periodic cylinder**: `PeriodicCylinderBasis`, `PeriodicCylinderBubble`, `PeriodicCylinderVector`
  and `PeriodicCylinderNormal`.
- **Vector Jacobi polynomials** on `[0,1]` (`VectorJacobi`, `vectorjacobip`).

## Example

The gradient of the weighted (vanishing on the boundary) disk basis for Fourier mode `m = 2`
is expressed in a curl-free vector basis, and the resulting stiffness matrix is diagonal:
```julia
julia> using DeRhamOrthogonalPolynomials, ClassicalOrthogonalPolynomials, StaticArrays

julia> W = Weighted(DiskBasis(2, 1));

julia> ∇ = grad(W);

julia> ∇[SVector(0.1, 0.2), 1:2]
2-element Vector{SVector{2, ComplexF64}}:
 [0.19600000000000006 + 0.37199999999999994im, -0.368 + 0.17400000000000002im]
 [-0.5675000000000001 - 0.9850000000000001im, 0.9550000000000001 - 0.40250000000000014im]

julia> (∇'∇)[1:4,1:4]
4×4 FillArrays.OffDiagonal{Float64, Vector{Float64}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}:
 3.14159   ⋅         ⋅        ⋅ 
  ⋅       8.37758    ⋅        ⋅ 
  ⋅        ⋅       14.1372    ⋅ 
  ⋅        ⋅         ⋅      20.1062
```
