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
            Diagonal(BlockedVector(vec(Fill(convert(T,π)/4,2)*permutedims(weightedgrammatrix(Jacobi{T}(0,1)).diag)), (axes(P,2),)))
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
    @assert P.𝐚 == (0,0) && W.P.𝐚 ∈ ((0,1), (1,0), (1,1))
    T = real(eltype(promote_type(eltype(P), eltype(W))))
    if iszero(m)
        # (1-)
        L = (Weighted(Jacobi{T}(0,1)) \  Weighted(Jacobi{T}(1,1)))/2
        BlockBroadcastMatrix{T}(Diagonal, unitblocks(iszero(W.P.𝐚[1]) ? Eye{T}(∞) : L), unitblocks(iszero(W.P.𝐚[2]) ? Eye{T}(∞) : L))
    else
        Weighted(VectorJacobi{T}((0,0), abs(m)-1)) \ Weighted(VectorJacobi{T}(W.P.𝐚, abs(m)-1))
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


# c(X) for c the coefficients in jacobi(0, b, 0..1) and X, of bandwidth w, the multiplication by t = r^2. Clenshaw assumes w = 1,
# so we pad c to give the right bandwidth
function _equivariantclenshaw(c, b, X, w)
    T = eltype(c)
    Clenshaw([c; zeros(T, (w-1)*(length(c)-1))], recurrencecoefficients(jacobi(0, b, UnitInterval{real(T)}()))..., T.(X), one(T))
end

"""
    equivariantmultiplication(V::DiskVector, g)

is the operator `M` such that `A * V = V * M`, where `A` has coefficients `g` as in `disk2diskmatrix`.
"""
function equivariantmultiplication(V::DiskVector{T}, g::AbstractVector) where T
    @assert V.𝐚 == (0,0)
    m = V.m
    c = paddeddata(g)
    c = Complex{T}.(pad(c, 4max(1, cld(length(c), 4))))
    α, δ, β, γ = c[1:4:end], c[2:4:end], c[3:4:end], c[4:4:end]
    if iszero(m)
        # on (p*[x,y], p*[-y,x]), A acts as [α+t*β γ*t-δ; δ+t*γ α-t*β]
        X = (jacobimatrix(Jacobi{T}(0,1)) + I)/2
        A, D = _equivariantclenshaw(α, 0, X, 1), _equivariantclenshaw(δ, 0, X, 1)
        B, C = X*_equivariantclenshaw(β, 2, X, 1), X*_equivariantclenshaw(γ, 2, X, 1)
        BlockBroadcastMatrix{Complex{T}}(hvcat, 2, unitblocks(A+B), unitblocks(C-D), unitblocks(C+D), unitblocks(A-B))
    else
        # ρ(π/2) = [0 -1; 1 0] is -im*sign(m) on type-1 elements and im*sign(m) on type-2 elements
        X₁, X₂ = equivariantjacobimatrix(Val(1), V), equivariantjacobimatrix(Val(2), V)
        A, D = _equivariantclenshaw(α, 0, X₁, 2), _equivariantclenshaw(δ, 0, X₁, 2)
        B, C = _equivariantclenshaw(β, 2, X₁, 2), _equivariantclenshaw(γ, 2, X₁, 2)
        s = sign(m)*im*one(T)
        A + X₂*B + Diagonal(fourierblockinterlace(Fill(-s,∞), Fill(s,∞)))*(D + X₂*C)
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

axes(P::AbstractDiskHodge{T}) where T = (Inclusion(UnitDisk{T}()), unitblocks(oneto(∞)))
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

@simplify function *(Pc::QuasiAdjoint{<:Any, <:DiskVector}, Q::DiskVector)
    P = parent(Pc)
    m = P.m
    @assert m == Q.m
    P == Q && return grammatrix(P)
    T = real(eltype(promote_type(eltype(P), eltype(Q))))
    V = DiskVector{T}(m)
    (V\P)' * grammatrix(V) * (V\Q)
end

@simplify function *(Nc::QuasiAdjoint{<:Any, <:Weighted{<:Any, <:DiskVector}}, Q::DiskVector)
    N = parent(Nc)
    m = Q.m
    @assert m == N.P.m
    N.P == Q && return weightedgrammatrix(Q)
    T = real(eltype(promote_type(eltype(N), eltype(Q))))
    V = DiskVector{T}(m)
    (V\N)' * grammatrix(V) * (V\Q)
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
curl(f::AbstractQuasiArray) = curl_layout(MemoryLayout(f), f)
curl_layout(::ExpansionLayout, f) = curl(basis(f)) * coefficients(f)
divergence(f::AbstractQuasiArray) = divergence_layout(MemoryLayout(f), f)
divergence_layout(::ExpansionLayout, f) = divergence(basis(f)) * coefficients(f)
function grad(W::Weighted{<:Any,<:DiskBasis{T}}) where {T}
    @assert W.P.a == 1
    Weighted(DiskCurlFree{T}(W.P.m)) * Diagonal(unitblocks(-(one(T):∞)))
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

"""
    diskrotmatrix(T, m)

gives the coefficients of rotating by 90°, that is,
`[0 -1; 1 0] * DiskVector(m, (a+1,a)) == DiskVector(m, (a,a+1)) * diskrotmatrix(T, m)`.
"""
function diskrotmatrix(::Type{T}, m) where T
    ax = axes(DiskVector{T}(m), 2)
    if iszero(m)
        # [x,y] ↦ [-y,x] and [-y,x] ↦ -[x,y], i.e., each block is [0 -1; 1 0]
        z, o = Zeros{T}(1,∞), Ones{T}(1,∞)
        _BandedBlockBandedMatrix(BlockedMatrix(Vcat(vec(Vcat(z, -o))', z, vec(Vcat(o, z))'), (OneTo(3), ax)), ax, (0,0), (1,1))
    else
        # 𝐞_r ↦ 𝐞_θ and 𝐞_θ ↦ -𝐞_r swaps the components, matching swapping 𝐚 in vectorjacobip
        Diagonal(Fill(-im*sign(m)*one(T), (ax,)))
    end
end

"""
    divergence(𝐏::DiskVector)

represents the divergence of `𝐏 = DiskVector(m, (a,a+1))`, expanded in `DiskBasis(m, a+1)`.
"""
function divergence(𝐏::DiskVector{T}) where T
    m,(a,a₂) = 𝐏.m, 𝐏.𝐚
    @assert a₂ == a+1
    # R*𝐐 == 𝐏*diskrotmatrix(T, m) for R = [0 -1; 1 0] and 𝐐 = DiskVector(m, (a+1,a)), where R^2 == -I means
    # inv(diskrotmatrix(T, m)) == -diskrotmatrix(T, m). Hence as div(R*𝐮) == -curl(𝐮) we have
    # div(𝐏) == div(R*𝐐) * inv(diskrotmatrix(T, m)) == curl(𝐐) * diskrotmatrix(T, m)
    C = curl(DiskVector{T}(m, (a+1,a)))
    basis(C) * (coefficients(C) * diskrotmatrix(T, m))
end


## These are built on ComplexZernike from MultivariateOrthogonalPolynomials.
# Columns are ordered m = 0, -1, 1, -2, 2, ….

_complexzernikemode(c::Integer) = isodd(c) ? c ÷ 2 : -(c ÷ 2)
_complexzernikecolumn(m::Integer) = m > 0 ? 2m+1 : max(1, -2m)

# total degree of the r-th element of DiskVector(m), or of Weighted(DiskVector(m, 𝐰)) for 𝐰 = (0,1), (1,0), (1,1)
_diskvectordegree(m, r, 𝐰=(0,0)) = iszero(m) ? 2((r-1) ÷ 2) + 1 + 2𝐰[2-isodd(r)] : 2(r ÷ 2) + abs(m) - 1 + 2max(𝐰...)

# total degree of the term with coefficient g[r] in disk2diskmatrix
_diskmatrixdegree(r) = 2((r+1) ÷ 4)

# plans compute coefficients in DiskVector(m, (a,a)) with ComplexZernike(0, a), then convert to 𝐚
function _diskvectorplanweight(𝐚, weighted=false)
    a₁,a₂ = 𝐚
    if weighted
        𝐚 ∈ ((0,1), (1,0), (1,1)) || error("Not implemented for Weighted(DiskVector) with 𝐚 = $𝐚")
        0
    elseif a₁ == a₂ ≥ -1
        max(a₁, 0)
    elseif abs(a₁-a₂) == 1 && min(a₁,a₂) ≥ -1
        max(a₁, a₂)
    else
        error("Not implemented for DiskVector with 𝐚 = $𝐚")
    end
end

# conversions from 𝐚 to (a,a) for each column
function _diskvectorconversions(T, 𝐚, weighted, N)
    a = _diskvectorplanweight(𝐚, weighted)
    !weighted && 𝐚 == (a,a) && return nothing
    Ñ = N ÷ 2 + 1
    if weighted
        [LowerTriangular((DiskVector{T}(m) \ Weighted(DiskVector{T}(m, 𝐚)))[1:2Ñ,1:2Ñ]) for m in _complexzernikemode.(1:4Ñ-1)]
    else
        [UpperTriangular((DiskVector{T}(m, (a,a)) \ DiskVector{T}(m, 𝐚))[1:2Ñ,1:2Ñ]) for m in _complexzernikemode.(1:4Ñ-1)]
    end
end

"""
    disk2diskvector(F₁, F₂)

converts the `ComplexZernike(a)` coefficients `F₁`, `F₂` of the components of a vector field to its coefficients
in `DiskVector(m, (a,a))`, with column `j` for `m = (-1)^iseven(j) * j ÷ 2`.
"""
function disk2diskvector(F₁::AbstractMatrix, F₂::AbstractMatrix)
    @assert size(F₁) == size(F₂)
    n, ncols = size(F₁)
    F₋, F₊ = pad(F₁ - im*F₂, n, ncols+4), pad(F₁ + im*F₂, n, ncols+4) # u₁ - im*u₂ and u₁ + im*u₂
    G = zeros(eltype(F₋), 2n, ncols+2) # a field of degree n has modes up to n+1
    for j in 1:ncols+2
        m = _complexzernikemode(j)
        if iszero(m)
            # u₁ ± im*u₂ is r*exp(±im*θ) for 𝐮 = r*𝐞_r and ±im*r*exp(±im*θ) for 𝐮 = r*𝐞_θ
            for k in 1:n
                A,B = F₋[k, 2], F₊[k, 3]
                G[2k-1,j] = (A + B)/2
                G[2k, j] = im*(A - B)/2
            end
        else
            # for s = sign(m), u₁ - im*s*u₂ is the type-1 part, of mode m-s, and u₁ + im*s*u₂ the type-2 part, of mode m+s
            A, B = m > 0 ? (F₋, F₊) : (F₊, F₋)
            c₁, c₂ = _complexzernikecolumn(m - sign(m)), _complexzernikecolumn(m + sign(m))
            for k in 1:n
                G[2k-1, j] = A[k, c₁]/2
                G[2k, j] = B[k, c₂]/2
            end
        end
    end
    G
end

disk2diskvector(F₁::ModalTrav, F₂::ModalTrav) = disk2diskvector(F₁.matrix, F₂.matrix)

"""
    DiskVectorPlan(N, 𝐚=(0,0); weighted=false)

transforms from `grid(P)` to the coefficients in `DiskVector(m, 𝐚)` for all `m`
"""
struct DiskVectorPlan{T,PLAN,CONV}
    plan::PLAN
    N::Int
    𝐚::NTuple{2,Int}
    weighted::Bool
    R::CONV
end

function DiskVectorPlan{T}(N::Integer, 𝐚::NTuple{2,Integer}; weighted::Bool=false) where T<:Real
    a = _diskvectorplanweight(𝐚, weighted)
    pl = plan_transform(ComplexZernike{Complex{T}}(0, a), (Block(N),))
    R = _diskvectorconversions(T, 𝐚, weighted, N)
    DiskVectorPlan{T,typeof(pl),typeof(R)}(pl, N, 𝐚, weighted, R)
end
DiskVectorPlan{T}(N::Integer, a::Integer=0; kws...) where T<:Real = DiskVectorPlan{T}(N, (a,a); kws...)
DiskVectorPlan(N::Integer, 𝐚=0; kws...) = DiskVectorPlan{Float64}(N, 𝐚; kws...)

grid(P::DiskVectorPlan{T}) where T = grid(ComplexZernike{Complex{T}}(), Block(P.N))

function *(P::DiskVectorPlan, F::AbstractMatrix{<:StaticVector{2}})
    G = disk2diskvector(P.plan * getindex.(F, 1), P.plan * getindex.(F, 2))
    for j in axes(G,2)
        m = _complexzernikemode(j)
        G[_diskvectordegree.(m, axes(G,1)) .> P.N-1, j] .= 0
        isnothing(P.R) || (G[:,j] = P.R[j] \ G[:,j])
        P.weighted && (G[_diskvectordegree.(m, axes(G,1), Ref(P.𝐚)) .> P.N-1, j] .= 0) # the weight raises the degree
    end
    G
end


## Inverse transforms

"""
    ComplexZernikeITransform{T}(N, a=0)

is the inverse of `plan_transform(ComplexZernike(0, a), (Block(N),))`, ignoring coefficients of degree above `2(N ÷ 2)`.
"""
struct ComplexZernikeITransform{T,PLAN}
    plan::PLAN
    N::Int
    a::Int
end

function ComplexZernikeITransform{T}(N::Integer, a::Integer=0) where T<:Real
    pl = MultivariateOrthogonalPolynomials.ZernikeITransform{T}(N, 0, a)
    ComplexZernikeITransform{T,typeof(pl)}(pl, N, a)
end
ComplexZernikeITransform(N::Integer, a::Integer=0) = ComplexZernikeITransform{Float64}(N, a)

grid(P::ComplexZernikeITransform{T}) where T = grid(ComplexZernike{Complex{T}}(), Block(P.N))
inv(P::ComplexZernikeITransform{T}) where T = plan_transform(ComplexZernike{Complex{T}}(0, P.a), (Block(P.N),))

function *(P::ComplexZernikeITransform{T}, F::AbstractMatrix) where T
    # the entries of F of degree at most 2Ñ-2, in the Ñ × (4Ñ-3) coefficient matrix of the transform
    Ñ = P.N ÷ 2 + 1
    C = zeros(Complex{T}, Ñ, 4Ñ-3)
    for c in axes(F,2), k in axes(F,1)
        2(k-1) + abs(_complexzernikemode(c)) ≤ 2Ñ-2 && (C[k,c] = F[k,c])
    end
    # undo MultivariateOrthogonalPolynomials._zernike2complex with α = 1/2, then the normalisation
    A = similar(C)
    A[:,1] = C[:,1]
    for j in 2:2:size(C,2)
        A[:,j] = im*(C[:,j+1] - C[:,j])
        A[:,j+1] = C[:,j] + C[:,j+1]
    end
    A ./= MultivariateOrthogonalPolynomials._zernikemodalconstant(T, size(A), 0, P.a)
    P.plan * ModalTrav(real(A)) + im * (P.plan * ModalTrav(imag(A)))
end
*(P::ComplexZernikeITransform, F::ModalTrav) = P * F.matrix

"""
    diskvector2disk(G)

is the inverse of `disk2diskvector`, converting the coefficients `G` in `DiskVector(m, (a,a))`, with column `j` for
`m = (-1)^iseven(j) * j ÷ 2`, to the `ComplexZernike(a)` coefficients `(F₁, F₂)` of the components.
"""
function diskvector2disk(G::AbstractMatrix)
    n,ncols = cld(size(G,1), 2), size(G,2)
    G = pad(G, 2n, ncols)
    F₋ = zeros(complex(eltype(G)), n, ncols+2) # u₁ - im*u₂
    F₊ = zeros(complex(eltype(G)), n, ncols+2) # u₁ + im*u₂
    for j in 1:ncols
        m = _complexzernikemode(j)
        if iszero(m)
            for k in 1:n
                cʳ,cᶿ = G[2k-1,j], G[2k,j]
                F₋[k,2] = cʳ - im*cᶿ
                F₊[k,3] = cʳ + im*cᶿ
            end
        else
            A,B = m > 0 ? (F₋,F₊) : (F₊,F₋)
            c₁,c₂ = _complexzernikecolumn(m - sign(m)), _complexzernikecolumn(m + sign(m))
            for k in 1:n
                A[k,c₁] = 2G[2k-1,j]
                B[k,c₂] = 2G[2k,j]
            end
        end
    end
    (F₋ + F₊)/2, im*(F₋ - F₊)/2
end

"""
    DiskVectorIPlan(N, 𝐚=(0,0); weighted=false)

is the inverse of `DiskVectorPlan(N, 𝐚; weighted)`, ignoring coefficients of degree above `2(N ÷ 2)`.
"""
struct DiskVectorIPlan{T,PLAN,CONV}
    plan::ComplexZernikeITransform{T,PLAN}
    𝐚::NTuple{2,Int}
    weighted::Bool
    R::CONV
end

function DiskVectorIPlan{T}(N::Integer, 𝐚::NTuple{2,Integer}; weighted::Bool=false) where T<:Real
    a = _diskvectorplanweight(𝐚, weighted)
    DiskVectorIPlan(ComplexZernikeITransform{T}(N, a), 𝐚, weighted, _diskvectorconversions(T, 𝐚, weighted, N))
end
DiskVectorIPlan{T}(N::Integer, a::Integer=0; kws...) where T<:Real = DiskVectorIPlan{T}(N, (a,a); kws...)
DiskVectorIPlan(N::Integer, 𝐚=0; kws...) = DiskVectorIPlan{Float64}(N, 𝐚; kws...)

grid(P::DiskVectorIPlan) = grid(P.plan)
inv(P::DiskVectorPlan{T}) where T =
    DiskVectorIPlan(ComplexZernikeITransform{T}(P.N, _diskvectorplanweight(P.𝐚, P.weighted)), P.𝐚, P.weighted, P.R)
inv(P::DiskVectorIPlan{T}) where T = DiskVectorPlan{T}(P.plan.N, P.𝐚; weighted=P.weighted)

function *(P::DiskVectorIPlan, G::AbstractMatrix)
    if !isnothing(P.R)
        # convert the columns to DiskVector(m, (a,a)), dropping the coefficients of degree above 2Ñ-2
        Ñ = P.plan.N ÷ 2 + 1
        𝐰 = P.weighted ? P.𝐚 : (0,0)
        G₀ = zeros(complex(eltype(G)), 2Ñ, 4Ñ-1)
        for j in axes(G,2), r in axes(G,1)
            _diskvectordegree(_complexzernikemode(j), r, 𝐰) ≤ 2Ñ-2 && (G₀[r,j] = G[r,j])
        end
        for j in axes(G₀,2)
            G₀[:,j] = P.R[j] * G₀[:,j]
        end
        G = G₀
    end
    F₁,F₂ = diskvector2disk(G)
    SVector.(P.plan * F₁, P.plan * F₂)
end


#####
# Equivariant matrix transforms
#####

"""
    disk2diskmatrix(F₁₁, F₁₂, F₂₁, F₂₂)

converts the `ComplexZernike()` coefficients `Fᵢⱼ` of the entries `Aᵢⱼ` of a matrix field `A` to the coefficients `g` of its
equivariant part `α(r^2)*I + δ(r^2)*ρ(π/2) + β(r^2)*Σ + γ(r^2)*ρ(π/2)*Σ`, with `Σ = [x^2-y^2 2x*y; 2x*y y^2-x^2]`:
`g[4k-3:4k]` are the `k`-th coefficients of `α` and `δ` in `jacobi(0, 0, 0..1)` and of `β` and `γ` in `jacobi(0, 2, 0..1)`.
"""
function disk2diskmatrix(F₁₁::AbstractMatrix, F₁₂::AbstractMatrix, F₂₁::AbstractMatrix, F₂₂::AbstractMatrix)
    @assert size(F₁₁) == size(F₁₂) == size(F₂₁) == size(F₂₂)
    n,ncols = size(F₁₁)
    # A = α*I + δ*ρ(π/2) + β*Σ + γ*ρ(π/2)*Σ has A₁₁ + A₂₂ = 2α, A₂₁ - A₁₂ = 2δ and (A₁₁ - A₂₂) ∓ im*(A₁₂ + A₂₁) = 2(β ∓ im*γ)*r^2*exp(∓2im*θ)
    F₋ = pad((F₁₁ - F₂₂ - im*(F₁₂ + F₂₁))/2, n, ncols+4)
    F₊ = pad((F₁₁ - F₂₂ + im*(F₁₂ + F₂₁))/2, n, ncols+4)
    c₋,c₊ = _complexzernikecolumn(-2), _complexzernikecolumn(2)
    g = zeros(eltype(F₋), 4n)
    for k in 1:n
        A,B = F₋[k,c₋], F₊[k,c₊]
        g[4k-3] = (F₁₁[k,1] + F₂₂[k,1])/2
        g[4k-2] = (F₂₁[k,1] - F₁₂[k,1])/2
        g[4k-1] = (A + B)/2
        g[4k] = im*(A - B)/2
    end
    g
end

disk2diskmatrix(F₁₁::ModalTrav, F₁₂::ModalTrav, F₂₁::ModalTrav, F₂₂::ModalTrav) = disk2diskmatrix(F₁₁.matrix, F₁₂.matrix, F₂₁.matrix, F₂₂.matrix)

"""
    diskmatrix2disk(g)

is the inverse of `disk2diskmatrix`, returning the `ComplexZernike()` coefficients `(F₁₁, F₁₂, F₂₁, F₂₂)` of the entries.
"""
function diskmatrix2disk(g::AbstractVector)
    n = cld(length(g), 4)
    g = pad(g, 4n)
    c₋,c₊ = _complexzernikecolumn(-2), _complexzernikecolumn(2)
    F₁₁,F₁₂,F₂₁,F₂₂ = ntuple(_ -> zeros(complex(eltype(g)), n, c₊), 4)
    for k in 1:n
        α,δ,β,γ = g[4k-3], g[4k-2], g[4k-1], g[4k]
        A,B = (β - im*γ)/2, (β + im*γ)/2
        F₁₁[k,1], F₁₂[k,1], F₂₁[k,1], F₂₂[k,1] = α, -δ, δ, α
        F₁₁[k,c₋], F₁₂[k,c₋], F₂₁[k,c₋], F₂₂[k,c₋] = A, im*A, im*A, -A
        F₁₁[k,c₊], F₁₂[k,c₊], F₂₁[k,c₊], F₂₂[k,c₊] = B, -im*B, -im*B, -B
    end
    F₁₁,F₁₂,F₂₁,F₂₂
end

"""
    DiskMatrixPlan(N)

transforms from `grid(P)` to the coefficients of the equivariant part, as in `disk2diskmatrix`. It is exact up to degree `N-1`.
"""
struct DiskMatrixPlan{T,PLAN}
    plan::PLAN
    N::Int
end

function DiskMatrixPlan{T}(N::Integer) where T<:Real
    pl = plan_transform(ComplexZernike{Complex{T}}(), (Block(N),))
    DiskMatrixPlan{T,typeof(pl)}(pl, N)
end
DiskMatrixPlan(N::Integer) = DiskMatrixPlan{Float64}(N)

grid(P::DiskMatrixPlan{T}) where T = grid(ComplexZernike{Complex{T}}(), Block(P.N))

function *(P::DiskMatrixPlan, F::AbstractMatrix{<:StaticMatrix{2,2}})
    g = disk2diskmatrix(P.plan * getindex.(F, 1, 1), P.plan * getindex.(F, 1, 2), P.plan * getindex.(F, 2, 1), P.plan * getindex.(F, 2, 2))
    g[_diskmatrixdegree.(eachindex(g)) .> P.N-1] .= 0
    g
end

"""
    DiskMatrixIPlan(N)

is the inverse of `DiskMatrixPlan(N)`, ignoring coefficients of degree above `2(N ÷ 2)`.
"""
struct DiskMatrixIPlan{T,PLAN}
    plan::ComplexZernikeITransform{T,PLAN}
end

DiskMatrixIPlan{T}(N::Integer) where T<:Real = DiskMatrixIPlan(ComplexZernikeITransform{T}(N))
DiskMatrixIPlan(N::Integer) = DiskMatrixIPlan{Float64}(N)

grid(P::DiskMatrixIPlan) = grid(P.plan)
inv(P::DiskMatrixPlan{T}) where T = DiskMatrixIPlan{T}(P.N)
inv(P::DiskMatrixIPlan{T}) where T = DiskMatrixPlan{T}(P.plan.N)

function *(P::DiskMatrixIPlan, g::AbstractVector)
    F₁₁,F₁₂,F₂₁,F₂₂ = diskmatrix2disk(g)
    SMatrix{2,2}.(P.plan * F₁₁, P.plan * F₂₁, P.plan * F₁₂, P.plan * F₂₂)
end
