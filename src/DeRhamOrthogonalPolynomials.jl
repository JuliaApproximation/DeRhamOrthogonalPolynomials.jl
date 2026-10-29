module DeRhamOrthogonalPolynomials

using ClassicalOrthogonalPolynomials, QuasiArrays, LinearAlgebra, StaticArrays, BlockArrays, HarmonicOrthogonalPolynomials, MultivariateOrthogonalPolynomials, LazyArrays, FillArrays, LazyBandedMatrices, InfiniteArrays, BlockBandedMatrices, ContinuumArrays, DomainSets, Infinities, ArrayLayouts


using BlockArrays: block, blockindex
import Base: getindex, axes, \, *, ==, OneTo, tail
import ClassicalOrthogonalPolynomials: orthogonalityweight, Weight, jacobimatrix, weightedgrammatrix, grammatrix, HalfWeighted, transform_ldiv, pad
using BandedMatrices: _BandedMatrix
using InfiniteArrays: OneToInf
using BlockBandedMatrices: _BandedBlockBandedMatrix
import ContinuumArrays: @simplify, ExpansionLayout, grid, AffineMap
using LazyArrays: ApplyLayout, arguments
using LazyBandedMatrices: _blockinterlace_axes, BlockBroadcastVector
using LinearAlgebra: norm_sqr
using HarmonicOrthogonalPolynomials: sphericalharmonicnormalization

export DiskVector, DiskZernikeWeight, DiskVectorWeight, grad, curl, rotgrad,
            DiskBasis, DiskCurlFree, DiskCoCurlFree
export VectorJacobi, vectorjacobip, VectorJacobiWeight, UnitDisk
export PeriodicCylinderBasis, PeriodicCylinderBubble, PeriodicCylinderVector, PeriodicCylinderNormal
export BallBasis, vectorzernike

include("vectorjacobi.jl")
include("disk.jl")
include("ball.jl")
include("periodiccylinder.jl")

end # module DeRhamOrthogonalPolynomials
