#####
# disk weights
#####

"""
    HyperZernikeWeight(a)

represents the weight (1-r^2)^α for r = norm(𝐱)
"""
struct HyperZernikeWeight{N,T} <: Weight{T}
    a::Int
end

HyperZernikeWeight{N}(a) where N = HyperZernikeWeight{N,Float64}(a)


axes(::HyperZernikeWeight{N,T}) where {N,T} = (Inclusion(StaticUnitBall{RadialCoordinate{T}, :closed}()),)

Base.unsafe_getindex(w::HyperZernikeWeight{N,T}, 𝐱::StaticVector{N}) where {N,T} = (one(T) - LinearAlgebra.norm_sqr(𝐱)) ^ w.a
function getindex(P::HyperZernikeWeight{N,T}, 𝐱::StaticVector{N}) where {N,T}
    @boundscheck checkbounds(P, 𝐱)
    Base.unsafe_getindex(P, 𝐱)
end

"""
    HyperBallVectorWeight((a,b))

represents the weight `T^a * N^b` where `T = I - 𝐱*𝐱'` and `N = w*I + 𝐱*𝐱'` are the matrix-weights that impose being tangential, normal on the boundary.
"""

struct HyperBallVectorWeight{N,T,N²} <: Weight{SMatrix{N,N,T,N²}}
    𝐚::NTuple{2,Int}
end



axes(::HyperBallVectorWeight{N,T}) where {N,T} = (Inclusion(StaticUnitBall{RadialCoordinate{T}, :closed}()),)
function getindex(P::HyperBallVectorWeight{N,T}, 𝐱::StaticVector{N}) where {N,T}
    @boundscheck checkbounds(P, 𝐱)
    Base.unsafe_getindex(P, 𝐱)
end

function Base.unsafe_getindex(w::HyperBallVectorWeight{N,T}, 𝐱::StaticVector{N}) where {N,T}
    a,b = w.𝐚
    (I - 𝐱*𝐱')^a * ((1-norm_sqr(𝐱))*I + 𝐱*𝐱')^b
end


include("harmonicpolynomials.jl")


abstract type AbstractHyperZernike{N,T} <: Basis{Complex{T}} end

axes(P::AbstractHyperZernike{N,T}) where {N,T} = (Inclusion(EuclideanUnitBall{N,T}()), unitblocks(oneto(∞)))
function getindex(P::AbstractHyperZernike{N}, 𝐱::StaticVector{N}, j::Int) where N
    @boundscheck checkbounds(P, 𝐱, j)
    Base.unsafe_getindex(P, 𝐱, j)
end

abstract type AbstractHyperBallVector{N,T} <: Basis{SVector{N,Complex{T}}} end


#####
# Disk
#####

const DiskZernikeWeight{T} = HyperZernikeWeight{2,T}

"""
    DiskVectorWeight((a,b))

represents

    [𝐞_r 𝐞_θ] * Diagonal([(1-r^2)^a, (1-r^2)^b])
"""
const DiskVectorWeight{T} = HyperBallVectorWeight{2,T,4}
DiskVectorWeight{T}() where T = DiskVectorWeight{T}((0,0))
DiskVectorWeight(𝐚...) = DiskVectorWeight{Float64}(𝐚...)

const AbstractDiskBasis{T} = AbstractHyperZernike{2,T}

axes(P::AbstractDiskBasis{T}) where {T} = (Inclusion(StaticUnitBall{RadialCoordinate{T}, :closed}()), unitblocks(oneto(∞)))


"""
    DiskBasis(m, a=0)

is orthogonal wrt to `(1-r^2)^a`.
"""
struct DiskBasis{T} <: AbstractDiskBasis{T}
    m::Int
    a::Int
end

DiskBasis{T}(m) where T = DiskBasis{T}(m, 0)
DiskBasis(m, a...) = DiskBasis{Float64}(m, a...)


==(A::DiskBasis, B::DiskBasis) = A.a == B.a && A.m == B.m
==(A::Weighted{<:Any, <:DiskBasis}, B::DiskBasis) = iszero(A.P.a) && A.P == B
==(B::DiskBasis, A::Weighted{<:Any, <:DiskBasis}) = A == B

function zernike(m, j::Int, λ, 𝐱::StaticVector{2})
    r² = LinearAlgebra.norm_sqr(𝐱)
    harmonicpolynomial(m, 𝐱) * jacobip(j, λ, abs(m), 2r²-1)
end

Base.unsafe_getindex(P::DiskBasis, 𝐱::StaticVector{2}, j::Int) = zernike(P.m, j-1, P.a, 𝐱)
function grammatrix(P::DiskBasis{T}) where T
    if P.a == 0
        unitblocks(convert(T,π) * weightedgrammatrix(Jacobi{T}(0, abs(P.m)))/convert(T,2)^(abs(P.m)+1))
    else
        Q = DiskBasis{T}(P.m)
        R = Q \ P
        R' * grammatrix(Q) * R
    end
end

orthogonalityweight(P::DiskBasis{T}) where {T} = DiskZernikeWeight{T}(P.a)



function \(P::DiskBasis, W::Weighted{<:Any, <:DiskBasis})
    m = P.m
    @assert m == W.P.m
    unitblocks((Jacobi(P.a, abs(m)) \ HalfWeighted{:a}(Jacobi(W.P.a, abs(m))))/2)
end

function \(Q::DiskBasis, P::DiskBasis)
    m = P.m
    @assert m == Q.m
    unitblocks(Jacobi(Q.a, abs(m)) \ Jacobi(P.a, abs(m)))
end


function grammatrix(W::Weighted{<:Any,<:DiskBasis{T}}) where T
    P = DiskBasis{T}(W.P.m)
    L = P\W
    L' * grammatrix(P) * L
end

@simplify function *(Wc::QuasiAdjoint{<:Any, <:Weighted{<:Any, <:DiskBasis}}, P::DiskBasis)
    W = parent(Wc)
    m = W.P.m
    @assert m == P.m
    T = real(promote_type(eltype(W), eltype(P)))
    Q = DiskBasis{T}(m)
    L = Q \ W
    R = Q \ P
    L' * grammatrix(Q) * R
end



struct DiskVector{T} <:AbstractHyperBallVector{2,T}
    m::Int
    𝐚::NTuple{2,Int}
end

DiskVector{T}(m) where T = DiskVector{T}(m, (0,0))
DiskVector(m, 𝐚...) = DiskVector{Float64}(m, 𝐚...)

axes(P::DiskVector{T}) where {T} = (Inclusion(StaticUnitBall{RadialCoordinate{T}, :closed}()), BlockedOneTo((1+iszero(P.m)):2:∞))

==(A::DiskVector, B::DiskVector) = A.m == B.m && A.𝐚 == B.𝐚
getindex(P::DiskVector, 𝐱::StaticVector{2}, j::Int) = P[𝐱, Block((j+1+!iszero(P.m))÷2)[min(j,mod(j-iszero(P.m),2)+1)]]
==(A::Weighted{<:Any,<:DiskVector}, B::DiskVector) = A.𝐚 == (0,0) && A.P == B
==(A::DiskVector, B::Weighted{<:Any,<:DiskVector}) = B.𝐚 == (0,0) && A == B.P

orthogonalityweight(P::DiskVector{T}) where {T} = DiskVectorWeight{T}(P.𝐚)

_angle(𝐱) = atan(reverse(𝐱)...)
_angle(𝐱::RadialCoordinate) = 𝐱.θ

function getindex(P::DiskVector, 𝐱::StaticVector{2}, Jk::BlockIndex)
    J,k = block(Jk), blockindex(Jk)
    m,(a,b) = P.m,P.𝐚
    n = Int(J)-1
    r,θ = norm(𝐱),_angle(𝐱)
    t = LinearAlgebra.norm_sqr(𝐱)
    x,y = 𝐱
    if iszero(m)
        isone(k) ? jacobip(n, a, 1, 2t-1) * SVector(x, y) : jacobip(n, b, 1, 2t-1) * SVector(-y, x)
    else
        𝐪 = vectorjacobip(n, k == 1 && Int(J) ≠ 1, (a,b), abs(m)-1, t)
        α = im*(1-2signbit(m))
        r^(abs(m)-1) * @SMatrix([cos(θ) -α*sin(θ); sin(θ) α*cos(θ)]) * 𝐪 * exp(im*m*θ)
    end
end

function grammatrix(P::DiskVector{T}) where T
    if P.m == 0
        if P.𝐚 == (0,0)
            Diagonal(vec(Fill(convert(T,π)/4,2)*permutedims(weightedgrammatrix(Jacobi{T}(0,1)).diag)))
        else
            a,b = P.𝐚
            P_a,P_b = Jacobi{T}(a,1),Jacobi{T}(b,1)
            M_a,M_b = HalfWeighted{:b}(P_a)' * P_a, HalfWeighted{:b}(P_b)' * P_b
            convert(T,π)/4 * BlockBroadcastMatrix{T}(Diagonal, unitblocks(M_a), unitblocks(M_b))
        end
    else
        convert(T,π)*grammatrix(HalfWeighted{:b}(VectorJacobi{T}(P.𝐚, abs(P.m)-1)))
    end
end


function grammatrix(W::Weighted{<:Any,<:DiskVector{T}}) where T
    P = DiskVector{T}(W.P.m)
    L = P\W
    L' * grammatrix(P) * L
end

function \(P::DiskVector, W::Weighted{<:Any,<:DiskVector})
    m = P.m
    @assert m == W.P.m
    @assert P.𝐚 == (0,0) && W.P.𝐚 == (0,1)
    T = real(eltype(promote_type(eltype(P), eltype(W))))
    if iszero(m)
        # (1-)
        L = (Weighted(Jacobi{T}(0,1)) \  Weighted(Jacobi{T}(1,1)))/2
        BlockBroadcastMatrix{T}(Diagonal, unitblocks(Eye{T}(∞)), unitblocks(L))
    else
        Weighted(VectorJacobi{T}((0,0), abs(m)-1)) \ Weighted(VectorJacobi{T}((0,1), abs(m)-1))
    end
end

function \(Q::DiskVector, P::DiskVector)
    m = Q.m
    @assert m == P.m
    T = real(eltype(promote_type(eltype(P), eltype(Q))))
    if iszero(m)
        R_1 = Jacobi{T}(Q.𝐚[1],1)\Jacobi{T}(P.𝐚[1],1)
        R_2 = Jacobi{T}(Q.𝐚[2],1)\Jacobi{T}(P.𝐚[2],1)
        BlockBroadcastMatrix{T}(Diagonal, unitblocks(R_1), unitblocks(R_2))
    else
        VectorJacobi{T}(Q.𝐚, abs(Q.m)-1) \ VectorJacobi{T}(P.𝐚, abs(P.m)-1)
    end
end
    
"""
    equivariantjacobimatrix(Val(1), P) / equivariantjacobimatrix(Val(2), P)

correspond to multiplication by r^2 * I or Σ, respectively
"""
function equivariantjacobimatrix(v::Val{k}, P::DiskVector{T}) where {T,k}
    m = P.m
    if iszero(m)
        # 2r^2 - 1 = τ => r^2 = (τ+1)/2
        X = unitblocks((jacobimatrix(Jacobi{T}(0,1)) + I)/2)
        BlockBroadcastMatrix{T}(Diagonal, X, (-1)^(k+1) * X)
    else
        jacobimatrix(v, VectorJacobi{T}((0,0), abs(m)-1))
    end
end


abstract type AbstractDiskHodge{T} <: Basis{SVector{2,Complex{T}}} end


struct DiskCurlFree{T} <: AbstractDiskHodge{T}
    m::Int
end

struct DiskCoCurlFree{T} <: AbstractDiskHodge{T}
    m::Int
end

DiskCurlFree(m) = DiskCurlFree{Float64}(m)
DiskCoCurlFree(m) = DiskCoCurlFree{Float64}(m)

axes(P::AbstractDiskHodge{T}) where T = (Inclusion(UnitDisk{T}()), oneto(∞))
orthogonalityweight(::AbstractDiskHodge{T}) where {T} = DiskVectorWeight{T}((0,1))
==(A::DiskCurlFree, B::DiskCurlFree) = A.m == B.m
==(A::DiskCoCurlFree, B::DiskCoCurlFree) = A.m == B.m
==(A::DiskCurlFree, B::DiskCoCurlFree) = false
==(A::DiskCoCurlFree, B::DiskCurlFree) = false
==(A::AbstractDiskHodge, B::DiskVector) = false
==(A::DiskVector, B::AbstractDiskHodge) = false
==(A::Weighted{<:Any,<:AbstractDiskHodge}, B::DiskVector) = false
==(A::DiskVector, B::Weighted{<:Any,<:AbstractDiskHodge}) = false


getindex(P::DiskCurlFree{T}, 𝐱::SVector{2}, k::Int) where T = iszero(P.m) ? 2DiskVector{T}(P.m, (0,1))[𝐱,Block(k)[1]]  : sum(DiskVector{T}(P.m, (0,1))[𝐱,Block(k+1)])

function getindex(P::DiskCoCurlFree{T}, 𝐱::SVector{2}, k::Int) where T
    m = P.m
    Q = DiskVector{T}(m, (0,1))
    if iszero(m)
        2Q[𝐱,Block(k)[2]]
    elseif isone(k)
        -2abs(m)*Q[𝐱,1]
    else
        𝐪 = DiskVector{T}(m, (0,1))[𝐱,Block(k)]
        (k-1)*(2k+abs(m))/(2*(2k+abs(m)-1)) * 𝐪[1] - 2(k+abs(m)-1)*𝐪[2]
    end
end


_block_lower_ones(T) = _BandedBlockBandedMatrix(
    Ones{T}((Base.OneTo(2), blockedrange(Ones{Int}(∞)))),
    BlockedOneTo(1:2:∞), (1,-1), (1,0))

_block_lower_firsttwo(T) = _BandedBlockBandedMatrix(
    Fill(convert(T,2), (Base.OneTo(1), blockedrange(Ones{Int}(∞)))),
    BlockedOneTo(2:2:∞), (0,0), (0,0))

function \(W::DiskVector{T}, P::DiskCurlFree{V}) where {T,V}
    @assert W.m == P.m && W.𝐚 == (0,1)
    if iszero(W.m)
        _block_lower_firsttwo(promote_type(T,V))
    else
        _block_lower_ones(promote_type(T,V))
    end
end


function \(W::DiskVector{T1}, P::DiskCoCurlFree{T2}) where {T1,T2}
    T = promote_type(T1,T2)
    j = 0:∞
    m = W.m
    @assert m == P.m && W.𝐚 == (0,1)
    if iszero(m)
        _BandedBlockBandedMatrix(Fill(convert(T,2), (oneto(1), BlockedOneTo(oneto(∞)))), (BlockedOneTo(2:2:∞), BlockedOneTo(oneto(∞))), (0,0), (1,-1))
    else
        _BandedBlockBandedMatrix(BlockedArray(
                Vcat(
                    Vcat(-2abs(m), ((j .+ 1) .* (2j .+ (abs(m)+4)) ./ (2*(2j .+ (abs(m)+3)))))',
                    -2*(j .+ abs(m))'),
                (Base.OneTo(2), blockedrange(Ones{Int}(∞)))), (BlockedOneTo(1:2:∞), BlockedOneTo(oneto(∞))), (0,0), (1,0))
    end
end

function \(A::DiskVector{V}, B::Weighted{<:Any,<:DiskCurlFree{T}}) where {T,V}
    @assert A.𝐚 == (0,0)
    DiskVector{V}(A.m, (0,1)) \ B.P # happens to be identical
end
\(A::Weighted{<:Any,<:DiskVector}, B::Weighted{<:Any,<:AbstractDiskHodge}) = A.P \ B.P
function \(A::DiskVector{V}, B::Weighted{<:Any,<:DiskCoCurlFree{T}}) where {T,V}
    N = Weighted(DiskVector{V}(A.m, (0,1)))
    (A\ N) * (N \ B)
end

function grammatrix(K::Weighted{<:Any,<:DiskCurlFree{T}}) where T
    if iszero(K.P.m)
        Diagonal(convert(T,π)*weightedgrammatrix(Jacobi{T}(0,1)).diag)
    else
        Diagonal(convert(T,π)*weightedgrammatrix(Jacobi(0,abs(K.P.m)-1)).diag[2:end] /convert(T,2)^(abs(K.P.m)-2))
    end
end

function weightedgrammatrix(K::DiskCurlFree{T}) where T
    a = abs(K.m)
    j = 0:∞
    convert(T,π)*Diagonal((2j .+ (a+2)) ./ ((j .+ (a+1)) .* (j .+ 1)))
end

function weightedgrammatrix(K::DiskCoCurlFree{T}) where T
    a = abs(K.m)
    iszero(a) && return convert(T,π)/2 * weightedgrammatrix(Jacobi{T}(1,1))
    j = 0:∞
    convert(T,π)*Diagonal((2j .+ a) .* (2j .+ (a+2)) ./ (2j .+ (a+1)))
end

function weightedgrammatrix(N::DiskVector{T}) where T
    @assert N.𝐚 == (0,1)
    a = convert(T,abs(N.m))
    if iszero(a)
         convert(T,π) * Diagonal(BlockBroadcastVector{T}(vcat, unitblocks(weightedgrammatrix(Jacobi{T}(0,1)).diag/4),
            unitblocks(weightedgrammatrix(Jacobi{T}(1,1)).diag/8)))
    else
        j = 0:∞
        convert(T,π) * Diagonal(fourierblockinterlace(
                 (2j .+ (a+2)) .* (2j .+ a) ./ ((j .+ a) .* 𝔯.(j, a-1)),
                16* (2j .+ (a+2)) .* (2j .+ (a+3)) ./ (4*(j .+ 1) .* 𝔯.(j .+ 1, a-1))))
    end
end

@simplify function *(Pc::QuasiAdjoint{<:Any, <:DiskBasis}, Q::DiskBasis)
    P = parent(Pc)
    m = P.m
    @assert m == Q.m

    N = DiskBasis{real(promote_type(eltype(P), eltype(Q)))}(m)
    (N\P)' * grammatrix(N) * (N\Q)
end

@simplify function *(Nc::QuasiAdjoint{<:Any, <:Weighted{<:Any, <:DiskVector}}, K::Weighted{<:Any, <:AbstractDiskHodge})
    N = parent(Nc)
    (Nc*N) * (N \ K)
end

@simplify function *(Nc::QuasiAdjoint{<:Any, <:Weighted{<:Any, <:Union{DiskVector,AbstractDiskHodge}}}, K::Union{DiskVector,AbstractDiskHodge})
    if Nc'.P == K
        weightedgrammatrix(K)
    else
        error("implement")
    end
end

function grammatrix(D::Weighted{<:Any,<:DiskCoCurlFree{T}}) where T
    V = DiskVector(D.P.m)
    (V\D)' * grammatrix(V) * (V\D)
end


    
####
# calculus
####

grad(f::AbstractQuasiArray) = grad_layout(MemoryLayout(f), f)
grad_layout(::ExpansionLayout, f) = grad(basis(f)) * coefficients(f)
function grad(W::Weighted{<:Any,<:DiskBasis{T}}) where {T}
    @assert W.P.a == 1
    Weighted(DiskCurlFree{T}(W.P.m)) * Diagonal(-(one(T):∞))
end

function grad(Z::DiskBasis{T}) where {T}
    (; a, m) = Z
    Q = DiskVector{T}(m, (a+1,a))
    if iszero(m)
        #   ∇P_j^(a, 0)(2r²-1) = e_r  d/dr P_j^(a, 0)(2r²-1) = 4 r P_j^(a, 0)'(2r^2-1) = 
        #   = e_r * 2r * (j+a+1) * P_{j-1}^(a+1, 1)(2r²-1)
        #   = [x,y] * 2 * (j+a+1) * P_{j-1}^(a+1, 1)(2r²-1)
        Q * _BandedBlockBandedMatrix(unitblocks((2*(convert(T,a) .+ (1:∞)))'), axes(Q,2), (-1,1), (0,0))
    else
        Q * laddergradmatrix(T, Z.a, abs(m))
    end
end

function disknormalcurlmatrix(T, m)
    j = 0:∞
    if iszero(m)
        transpose(_BandedBlockBandedMatrix(unitblocks(-convert(T,2)*j)', (BlockedOneTo(2:2:∞), BlockedOneTo(oneto(∞))), (-1,1), (1,-1)))
    else
        d = -2*(2j .+ abs(m)) .* (2j .+ (abs(m)+1)) .* (2j .+ (abs(m)+2)) ./ 𝔯.(j,abs(m)-1) .* (im*sign(m))
        transpose(_BandedBlockBandedMatrix(BlockedArray(
                        Vcat(transpose(Vcat(d[1],-d[2:end])), transpose(d)),
                        (Base.OneTo(2), blockedrange(Ones{Int}(∞)))), 
                        (BlockedOneTo(1:2:∞), BlockedOneTo(oneto(∞))), (0,0), (1,0)))
    end
end


function curl(W::Weighted{<:Any,<:DiskVector{T}}) where T
    @assert W.P.𝐚 == (0,1)
    ApplyQuasiArray(*, DiskBasis{T}(W.P.m), disknormalcurlmatrix(T, W.P.m))
end
function curl(W::Weighted{<:Any,<:DiskCoCurlFree{T}}) where T
    j = 0:∞
    m = W.P.m
    if iszero(m)
        ApplyQuasiArray(*, DiskBasis{T}(m), 
            _BandedMatrix(convert(T,-4) * (1:∞)', ∞, 1, -1))
    else
        ApplyQuasiArray(*, DiskBasis{T}(m), 
            Diagonal((convert(T,2)j .+ abs(m)) .* (convert(T,2)j .+ (abs(m)+2)) * (im*sign(m))))
    end
end

function curl(𝐏::DiskVector{T}) where {T}
    m,(a₁,a) = 𝐏.m, 𝐏.𝐚
    @assert a₁ == a+1
    P = DiskBasis{T}(m, a+1)
    if iszero(m)
        P * _BandedBlockBandedMatrix(unitblocks(convert(T,2):2:∞)', axes(𝐏,2), (0,0), (1,-1))'
    else
        P * ((sign(m)im) * laddercurlmatrix(T, a, abs(m)-1))
    end
end


rotgrad(P::DiskBasis{T}) where T = DiskCoCurlFree{T}(P.m) * (iszero(P.m) ? _BandedMatrix((1:∞)', ∞, -1, 1) : im*sign(P.m)*Eye{T}(∞))