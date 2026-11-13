module DeRhamOrthogonalPolynomials

using ClassicalOrthogonalPolynomials, QuasiArrays, LinearAlgebra, StaticArrays, BlockArrays, HarmonicOrthogonalPolynomials, MultivariateOrthogonalPolynomials, LazyArrays, FillArrays, LazyBandedMatrices, InfiniteArrays, BlockBandedMatrices, ContinuumArrays, DomainSets, Infinities, ArrayLayouts
using MultivariateOrthogonalPolynomials: ModalTrav
using ClassicalOrthogonalPolynomials: Clenshaw, recurrencecoefficients
using LazyArrays: paddeddata


using BlockArrays: block, blockindex, findblockindex
import Base: getindex, axes, \, *, ==, OneTo, tail, inv
import ClassicalOrthogonalPolynomials: orthogonalityweight, Weight, jacobimatrix, weightedgrammatrix, grammatrix, HalfWeighted, transform_ldiv, pad
using BandedMatrices: _BandedMatrix, BandedMatrix
using InfiniteArrays: OneToInf
using BlockBandedMatrices: _BandedBlockBandedMatrix
import ContinuumArrays: @simplify, ExpansionLayout, grid, AffineMap
using LazyArrays: ApplyLayout, arguments
using LazyBandedMatrices: _blockinterlace_axes, BlockBroadcastVector
using LinearAlgebra: norm_sqr
using HarmonicOrthogonalPolynomials: sphericalharmonicnormalization

export DiskVector, DiskMatrix, DiskVectorPlan, DiskVectorIPlan, disk2diskvector, diskvector2disk, ComplexZernikeITransform, DiskZernikeWeight, DiskVectorWeight, grad, curl, divergence, rotgrad,
            DiskBasis, DiskCurlFree, DiskWeightedCurlFree, DiskCoWeightedCurlFree
export DiskMatrixPlan, DiskMatrixIPlan, disk2diskmatrix, diskmatrix2disk, equivariantmultiplication
export VectorJacobi, vectorjacobip, VectorJacobiWeight, UnitDisk
export PeriodicCylinderBasis, PeriodicCylinderVector
export BallBasis, vectorzernike

include("vectorjacobi.jl")
include("disk.jl")
include("ball.jl")
include("periodiccylinder.jl")
include("transforms.jl")

end # module DeRhamOrthogonalPolynomials
