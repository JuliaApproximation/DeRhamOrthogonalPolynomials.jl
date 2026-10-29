periodiccylinder(::Type{T}) where T = cartesianproduct(UnitDisk{T}(), ℝ)

struct PeriodicCylinderWeight{T,WEIGHT} <: Weight{T}
    disk::WEIGHT
end

PeriodicCylinderWeight{T}(diskweight) where T = PeriodicCylinderWeight{T,typeof(diskweight)}(diskweight)
PeriodicCylinderWeight(diskweight) = PeriodicCylinderWeight{eltype(diskweight)}(diskweight)
axes(::PeriodicCylinderWeight{T}) where T = (Inclusion(periodiccylinder(T)),)

Base.unsafe_getindex(w::PeriodicCylinderWeight{T}, (x,y,z)::SVector{3}) where T = w.disk[SVector(x,y)]
function getindex(P::PeriodicCylinderWeight{T}, 𝐱::SVector{3}) where {T}
    @boundscheck checkbounds(P, 𝐱)
    Base.unsafe_getindex(P, 𝐱)
end

struct PeriodicCylinderMatrixWeight{T,VECTOR,SCALAR} <: Weight{SMatrix{3,3,T,9}}
    vector::VECTOR
    scalar::SCALAR
end

PeriodicCylinderMatrixWeight{T}(vector, scalar) where T = PeriodicCylinderMatrixWeight{T,typeof(vector), typeof(scalar)}(vector, scalar)
PeriodicCylinderMatrixWeight(vector, scalar) = PeriodicCylinderMatrixWeight{promote_type(eltype(eltype(vector)), eltype(scalar))}(vector, scalar)
axes(::PeriodicCylinderMatrixWeight{T}) where T = (Inclusion(periodiccylinder(T)),)

Base.unsafe_getindex(w::PeriodicCylinderMatrixWeight{T}, (x,y,z)::SVector{3}) where T =
    [[w.vector[SVector(x,y)]; SVector(zero(T),zero(T))'] SVector(zero(T),zero(T), w.scalar[SVector(x,y)])]
function getindex(P::PeriodicCylinderMatrixWeight{T}, 𝐱::SVector{3}) where {T}
    @boundscheck checkbounds(P, 𝐱)
    Base.unsafe_getindex(P, 𝐱)
end


struct PeriodicCylinderBasis{T, DISK} <: Basis{Complex{T}}
    disk::DISK
    k::Int
end
PeriodicCylinderBasis{T}(basis::AbstractQuasiMatrix, k::Integer) where T = PeriodicCylinderBasis{T, typeof(basis)}(basis, k)
PeriodicCylinderBasis(basis::AbstractQuasiMatrix, k::Integer) = PeriodicCylinderBasis{real(eltype(basis))}(basis, k)
PeriodicCylinderBasis(m::Integer, k::Integer) = PeriodicCylinderBasis(DiskBasis(m), k)

axes(P::PeriodicCylinderBasis{T}) where T = (Inclusion(periodiccylinder(T)), oneto(∞))

function getindex(P::PeriodicCylinderBasis, 𝐱::SVector{3}, j::Int)
    @boundscheck checkbounds(P, 𝐱, j)
    Base.unsafe_getindex(P, 𝐱, j)
end


==(A::PeriodicCylinderBasis, B::PeriodicCylinderBasis) = A.disk == B.disk && A.k == B.k

diskbasis(P::PeriodicCylinderBasis) = P.disk
Base.unsafe_getindex(P::PeriodicCylinderBasis{T}, (x,y,z)::SVector{3}, j::Int) where T = exp(im*P.k*z) * diskbasis(P)[SVector(x,y), j]


grammatrix(P::PeriodicCylinderBasis{T}) where T = 2convert(T,π) * grammatrix(diskbasis(P))
grammatrix(W::Weighted{<:Any, <:PeriodicCylinderBasis{T}}) where T = 2convert(T,π) * grammatrix(Weighted(diskbasis(W.P)))

orthogonalityweight(P::PeriodicCylinderBasis) = PeriodicCylinderWeight(orthogonalityweight(P.disk))

function \(P::PeriodicCylinderBasis, W::Weighted{<:Any, <:PeriodicCylinderBasis})
    @assert P.k == W.P.k
    diskbasis(P) \ Weighted(diskbasis(W.P))
end

function \(Q::PeriodicCylinderBasis, P::PeriodicCylinderBasis)
    @assert Q.k == P.k
    diskbasis(Q) \ diskbasis(P)
end


######
# Vector
######

struct PeriodicCylinderVector{T, DISKBas, DISKVec}  <: Basis{SVector{3,Complex{T}}}
    scalar::DISKBas
    vector::DISKVec
    k::Int
    function PeriodicCylinderVector{T, DISKBas, DISKVec}(scalar, vector, k) where {T, DISKBas, DISKVec}
        @assert scalar.m == vector.m
        new{T,DISKBas,DISKVec}(scalar, vector, k)
    end
end


PeriodicCylinderVector{T}(scalar, vector, k) where T = PeriodicCylinderVector{T, typeof(scalar), typeof(vector)}(scalar, vector, k)
PeriodicCylinderVector(scalar, vector, k) = PeriodicCylinderVector{promote_type(real(eltype(scalar)),real(eltype(eltype(vector))))}(scalar, vector, k)
PeriodicCylinderVector(m, k) = PeriodicCylinderVector(DiskBasis(m), DiskVector(m), k)

diskvector(P::PeriodicCylinderVector) = P.vector
diskbasis(P::PeriodicCylinderVector) = P.scalar

orthogonalityweight(P::PeriodicCylinderVector) = PeriodicCylinderMatrixWeight(orthogonalityweight(P.vector), orthogonalityweight(P.scalar))

_cylinder_vector_axes(m) = _blockinterlace_axes(axes(DiskVector(m),2), unitblocks(axes(DiskBasis(m),2)))
axes(P::PeriodicCylinderVector{T}) where {T} = (Inclusion(periodiccylinder(real(T))), _cylinder_vector_axes(P.scalar.m))

==(A::PeriodicCylinderVector, B::PeriodicCylinderVector) = A.scalar == B.scalar && A.vector == B.vector && A.k == B.k
getindex(P::PeriodicCylinderVector, 𝐱::SVector{3}, j::Int) = P[𝐱, findblockindex(axes(P,2), j)]

function getindex(P::PeriodicCylinderVector, 𝐱::SVector{3}, Jk::BlockIndex)
    x,y,z = 𝐱
    J,k = Int(block(Jk)), blockindex(Jk)
    if isodd(J)
        SVector(diskvector(P)[SVector(x,y),Block((J+1)÷2)[k]]..., 0)
    else
        SVector(0,0, diskbasis(P)[SVector(x,y),J ÷ 2])
    end * exp(im*P.k*z)
end



grammatrix(P::PeriodicCylinderVector{T}) where T = 2convert(T,π) * blockinterlace(grammatrix(diskvector(P)), unitblocks(grammatrix(diskbasis(P))))
grammatrix(W::Weighted{<:Any,<:PeriodicCylinderVector{T}}) where T = 2convert(T,π) * blockinterlace(grammatrix(Weighted(diskvector(W.P))), unitblocks(grammatrix(Weighted(diskbasis(W.P)))))


function \(A::PeriodicCylinderVector, B::PeriodicCylinderVector)
    R_s = diskbasis(A) \ diskbasis(B)
    R_v = diskvector(A) \ diskvector(B)
    blockinterlace(R_v, unitblocks(R_s))
end

function \(A::PeriodicCylinderVector, W::Weighted{<:Any,<:PeriodicCylinderVector})
    B = W.P
    L_s = diskbasis(A) \ Weighted(diskbasis(B))
    L_v = diskvector(A) \ Weighted(diskvector(B))
    blockinterlace(L_v, unitblocks(L_s))
end
