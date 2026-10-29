"""
    vectorjacobip(n, ν, (a₁, a₂), b, t)

Evaluate the degree-`n` vector OP for weight `diag((1-t)^a₁, (1-t)^a₂) t^b` on [0,1].
`ν == 0` selects the type-1 polynomial, `ν == 1` selects type-2.
"""
function vectorjacobip(n::Int, ν, (a₁,a₂)::NTuple{2,Int}, b::Int, t)
    if a₁ == a₂
        diagonalvectorjacobip(n, ν, a₁, b, t)
    elseif a₁ ≥ -1 && a₂ == a₁ + 1
        normalvectorjacobip(n, ν, a₁, b, t)
    elseif a₂ ≥ -1 && a₁ == a₂ + 1
        q = normalvectorjacobip(n, ν, a₂, b, t)
        SVector(q[2], q[1])
    else
        error("not implemented")
    end
end

function diagonalvectorjacobip(n, ν, a, b, t)
    if ν == 1
        t * jacobip(n-1, a, b+2, 2t-1) * SVector(1,-1)
    else
        jacobip(n, a, b, 2t-1) * SVector(1,1)
    end
end

𝔯(n,a,b) = 10n^2+n*(12a+13b+19)+4*(a+b+1)*(a+b+2)
𝔯(n, b) = 𝔯(n, 0, b)

function normalvectorjacobip(n, ν, b, t)
    τ = 2t-1
    𝐯 = SVector(-2t*(n+1)*jacobip(n,0,b+2,τ) + (b+1) * (1-t)*jacobip(n,1,b+1,τ), (b+1)*jacobip(n,1,b+1,τ))
    𝐪¹ = (2n+b+1)/((n+1)*𝔯(n,b)) * ((2n+b+2)*(2n+b+3)* SVector((1-t), 1) * jacobip(n,1,b,τ) - (n+b+2) * 𝐯)
    if ν == 1
        𝐯̃ = SVector(-2t*n*jacobip(n-1,0,b+2,τ) + (b+1) * (1-t)*jacobip(n-1,1,b+1,τ), (b+1)*jacobip(n-1,1,b+1,τ))
        -(𝐯̃/n + 𝐪¹)
    else
        𝐪¹
    end
end

function normalvectorjacobip(n, ν, a, b, t)
    n == 0 && a == -1 && return SVector(one(t)/2, one(t)/2)
    τ = 2t-1
    𝐯 = (2n+a+b+3)/(2n+2a+b+3) * SVector(-2t*(n+a+1)*jacobip(n,a,b+2,τ) + (b+1)*(1-t)*jacobip(n,a+1,b+1,τ), (b+1)*jacobip(n,a+1,b+1,τ))
    𝐪¹ = (2n+a+b+1)*(2n+2a+b+3) / ((n+a+1)*(2n+a+b+3) * 𝔯(n,a,b)) * ((2n+a+b+2)*(2n+a+b+3)*jacobip(n,a+1,b,τ) * SVector(1-t,1) - (n+a+b+2)*𝐯)
    if ν == 1
        𝐯̃ = (2(n-1)+a+b+3)/(2(n-1)+2a+b+3) * SVector(-2t*((n-1)+a+1)*jacobip((n-1),a,b+2,τ) + (b+1)*(1-t)*jacobip((n-1),a+1,b+1,τ), (b+1)*jacobip((n-1),a+1,b+1,τ))
        -(𝐯̃/n + 𝐪¹)
    else
        𝐪¹
    end
end

"""
    VectorJacobiWeight((a₁, a₂), b)

represents
```math
    Diagonal([(1-t)^a₁, (1-t)^a₂])t^b on 0..1
```
"""
struct VectorJacobiWeight{T} <: Weight{Diagonal{SVector{2,T}}}
    𝐚::NTuple{2,Int}
    b::Int
end

VectorJacobiWeight(𝐚, b) = VectorJacobiWeight{Float64}(𝐚, b)

axes(::VectorJacobiWeight) = (Inclusion(UnitInterval{real(T)}()),)

function getindex(W::VectorJacobiWeight{T}, t) where T
    (a₁, a₂),b = W.𝐚,W.b
    Diagonal(SVector{2,T}((1-t)^a₁, (1-t)^a₂) * t^b)
end

"""
    VectorJacobi((a₁, a₂), b)

gives the vector-valued quasi-matrix corresponding to:
```math
    Diagonal([(1-t)^a₁, (1-t)^a₂])t^b on 0..1
```
"""
struct VectorJacobi{T} <: Basis{SVector{2,T}}
    𝐚::NTuple{2,Int}
    b::Int
end

==(P::VectorJacobi, Q::VectorJacobi) = P.𝐚 == Q.𝐚 && P.b == Q.b

const VECJACOBIAXES2 = BlockedOneTo(1:2:∞)

# Banded operators on VectorJacobi are built from a BlockHcat whose column axis has blocklasts
# Vcat(1, 3:2:∞). Give the data the canonical VECJACOBIAXES2 so that lazy products with other
# operators see identical block axes, which avoids comparing the two infinite blocklasts entrywise.
function _vectorjacobibanded(data::BlockHcat{T}, l, u) where T
    ax = (axes(data,1), VECJACOBIAXES2)
    _BandedMatrix(BlockHcat{T,typeof(data.arrays),typeof(ax)}(data.arrays, ax), VECJACOBIAXES2, l, u)
end

axes(::VectorJacobi{T}) where T = (Inclusion(UnitInterval{real(T)}()), VECJACOBIAXES2)

VectorJacobi(𝐚, b) = VectorJacobi{Float64}(𝐚, b)

orthogonalityweight(V::VectorJacobi{T}) where T = VectorJacobiWeight{T}(V.𝐚, V.b)

function getindex(P::VectorJacobi, t::Number, Jj::BlockIndex)
    J,j = Int(block(Jj)),blockindex(Jj)
    vectorjacobip(J-1, j == 1 && J ≠ 1, P.𝐚, P.b, t)
end

getindex(P::VectorJacobi, t::Number, j::Int) = P[t, Block((j+2)÷2)[mod(j,2)+1]]

fourierblockinterlace(a, b) = BlockedVector(vec(Vcat(permutedims(a), permutedims(b))), (VECJACOBIAXES2,))


@simplify function *(Ac::QuasiAdjoint{<:Any,<:VectorJacobi}, B::Weighted{<:Any,<:VectorJacobi})
    P = Ac'
    (B.P\P)' *  weightedgrammatrix(B.P)
end


function weightedgrammatrix(P::VectorJacobi{T}) where T
    (a,a₂),b = P.𝐚, P.b
    @assert a == a₂
    W_A = weightedgrammatrix(Jacobi{T}(a,b))/convert(T,2)^(a+b)
    W_B = weightedgrammatrix(Jacobi{T}(a,b+2))/convert(T,2)^(a+b+2)
    Diagonal(fourierblockinterlace(W_A.diag, W_B.diag))
end

function grammatrix(Q::VectorJacobi{T}) where T
    @assert Q.b == 0
    P = VectorJacobi{T}((0,0), 0)
    R = P\Q
    R' * weightedgrammatrix(P) * R
end

function grammatrix(Q::HalfWeighted{:b, SVector{2,T}, VectorJacobi{T}}) where T
    P = VectorJacobi{T}((0,0), Q.P.b)
    R = P\Q.P
    R' * weightedgrammatrix(P) * R
end


#####
# Conversion
#####

function \(W_P::Weighted{<:Any,<:VectorJacobi}, W::Weighted{<:Any,<:VectorJacobi})
    P = W_P.P
    b = P.b
    @assert P.𝐚 == (0,0)
    @assert b == W.P.b
    @assert W.P.𝐚 == (0,1)

    #  W * 𝐩(m,j,τ) == 2(j+1) / (2j+m+1) * 𝐏(m,j,τ) + (j+1)*(j+m+1) / ((2j+m+1)*(2j+m+2)) * 𝐐(m,j,τ) - 2 * (j+1)^2/((2j+m+1)*(2j+m+2)) * 𝐏(m,j+1,τ)
    # W * 𝐪(m,j,τ) == -j/2 * 𝐐(m,j-1,τ) - (j*(j+m)*(2+6j+3m))/(10j^2+4m*(1+m)+j*(6+13m)) * 𝐏(m,j,τ) + (2j+m)*j*(j+m+1)/(2*(10j^2+4m*(1+m)+j*(6+13m))) * 𝐐(m,j,τ)   - (2j+m)* (j+1)*j/(10j^2+4m*(1+m)+j*(6+13m)) * 𝐏(m,j+1,τ)
    T = real(eltype(promote_type(eltype(P), eltype(W))))

    n = 1:∞
    𝔯v = BroadcastVector(𝔯, n, b)
    κ = BroadcastVector((n,b) -> (2n+(b+1)) / 𝔯(n,b), n, b)
    _vectorjacobibanded(
        BlockHcat(unitblocks([(b+3)/(4*(b+2)), one(T)/4, -1/(4*(b+2)), 0]),
        BlockVcat(
            BlockBroadcastVector{T}(vcat, 
                unitblocks(Ones{T}(∞)), 
                unitblocks(κ .* (2n.+(b+3))))',
            BlockBroadcastVector{T}(vcat, 
                unitblocks((n.+(b+1)) .* (6n.+(3b+5)) ./ 𝔯v),
                unitblocks(κ .* (n.+(b+2))))',
            BlockBroadcastVector{T}(vcat,
                unitblocks(-(n.+(b+2)) .* κ),
                unitblocks(κ .* (-(n.+1))))',
            BlockBroadcastVector{T}(vcat, 
                unitblocks((n.+1) .* κ),
                unitblocks(Zeros{T}(∞)))'
    )
    ), 3, 0)
end

function \(Q::VectorJacobi, P::VectorJacobi)
    T = real(eltype(promote_type(eltype(P), eltype(Q))))
    Q == P && return Eye{T}((axes(P,2),))
    b = Q.b
    @assert b == P.b

    n = 1:∞
    if P.𝐚[1] + 1 == Q.𝐚[1] == Q.𝐚[2] == P.𝐚[2]
        a = P.𝐚[1]
        𝔯v = BroadcastVector(𝔯, n, a, b)
        κ = BroadcastVector((n,b) -> (2n+(b+1))^2, n, b) # avoid bug in Julia v1.12
        _vectorjacobibanded(
            BlockHcat(unitblocks([0,0,0,one(T)/2]),
            BlockVcat(
            BlockBroadcastVector{T}(vcat,
                unitblocks(Zeros{T}(∞)),
                unitblocks(-(2n .+ (2a+b+3)) .* (n .+ a) ./ (2𝔯v) ))',
            BlockBroadcastVector{T}(vcat,
                unitblocks(-(2 * (n .+ a) .* ((1+b) .+ n) .* ((1+a+b) .+ 2n) .* ((2+a+b) .+ 2n)) ./ (n .* ((1+2a+b) .+ 2n) .*𝔯v )),
                unitblocks(-(2n .+ (2a+b+3)) .* (n .+ b) ./ (2𝔯v)))',
            BlockBroadcastVector{T}(vcat,
                unitblocks(  -((2*(b .+ n) .* ((1+b) .+ n) .* ((1+a+b) .+ 2n) .* ((2+a+b) .+ 2n)) ./ (n .* ((1+2a+b) .+ 2n) .* 𝔯v))),
                unitblocks(-(n .+ (a+b+2)) .* (6n .+ (4a+3b+5)) ./ (2𝔯v)))',
            BlockBroadcastVector{T}(vcat,
                unitblocks((2*((2+a+b) .+ n) .* ((1+a+b) .+ 2n) .* ((2+a+b) .+ 2n)) ./ (n .* 𝔯v)),
                unitblocks( Ones{T}(∞)/2 ))'
            )), 0, 3)
    elseif P.𝐚[2] + 1 == Q.𝐚[1] == Q.𝐚[2] == P.𝐚[1]
        a = P.𝐚[2]
        𝔯v = BroadcastVector(𝔯, n, a, b)
        κ = BroadcastVector((n,b) -> (2n+(b+1))^2, n, b) # avoid bug in Julia v1.12
        _vectorjacobibanded(
            BlockHcat(unitblocks([0,0,0,one(T)/2]),
            BlockVcat(
            BlockBroadcastVector{T}(vcat,
                unitblocks(Zeros{T}(∞)),
                unitblocks((2n .+ (2a+b+3)) .* (n .+ a) ./ (2𝔯v) ))',
            BlockBroadcastVector{T}(vcat,
                unitblocks((2 * (n .+ a) .* ((1+b) .+ n) .* ((1+a+b) .+ 2n) .* ((2+a+b) .+ 2n)) ./ (n .* ((1+2a+b) .+ 2n) .*𝔯v )),
                unitblocks(-(2n .+ (2a+b+3)) .* (n .+ b) ./ (2𝔯v)))',
            BlockBroadcastVector{T}(vcat,
                unitblocks(  -((2*(b .+ n) .* ((1+b) .+ n) .* ((1+a+b) .+ 2n) .* ((2+a+b) .+ 2n)) ./ (n .* ((1+2a+b) .+ 2n) .* 𝔯v))),
                unitblocks((n .+ (a+b+2)) .* (6n .+ (4a+3b+5)) ./ (2𝔯v)))',
            BlockBroadcastVector{T}(vcat,
                unitblocks(-(2*((2+a+b) .+ n) .* ((1+a+b) .+ 2n) .* ((2+a+b) .+ 2n)) ./ (n .* 𝔯v)),
                unitblocks( Ones{T}(∞)/2 ))'
            )), 0, 3)
    elseif P.𝐚 == (0,0) && Q.𝐚 == (0,1)
        𝔯v = BroadcastVector(𝔯, n, b)
        κ = BroadcastVector((n,b) -> (2n+(b+1))^2, n, b) # avoid bug in Julia v1.12
        _vectorjacobibanded(
            BlockHcat(unitblocks([0,0,0,T(2)]),
            BlockVcat(
            BlockBroadcastVector{T}(vcat,
                unitblocks(Zeros{T}(∞)),
                unitblocks((n.-1) .* n ./ (2*(2n.+b) .* (2n.+(b+1)))))',
            BlockBroadcastVector{T}(vcat,
                unitblocks(-(n.-1) .* (n.+(b+1)) ./ (2*(2n.+b).*(2n.+(b+1)))),
                unitblocks(-2n .* (n.+b) ./ κ))',
            BlockBroadcastVector{T}(vcat,
                unitblocks(2*(n.+b) .* (n.+(b+1)) ./ κ),
                unitblocks(n .* (n.+(b+1)) .* (6n.+(3b+5)) ./ (2*κ .* (2n.+(b+2)))))',
            BlockBroadcastVector{T}(vcat,
                unitblocks(n .* 𝔯v ./ (2 * κ .* (2n.+(b+2)))),
                unitblocks(2*(n.+(b+1)) ./ (2n .+ (b+1)) ))'
            )), 0, 3)
    else
        error("Not implemented for $Q \\ $P")
    end
end



#####
# Jacobi matrices
#####

function jacobimatrix(::Val{1}, P::VectorJacobi{T}) where T
    @assert P.𝐚 == (0,0)
    b = P.b
    n = 0:∞
    _vectorjacobibanded(
        BlockHcat(unitblocks([0,0,b /(2*(b+2)) + one(T)/2,0, one(T) / (b+2)]),
        BlockVcat(
        BlockBroadcastVector{T}(vcat,
            unitblocks(n .* (n .+ (b+2)) ./ ((2n .+ (b+2)) .* (2n .+ (b+3)))),
            unitblocks((n .+ 1) .* (n .+ (b+1)) ./ ((2n .+ (b+2)) .* (2n .+ (b+3)))))',
        Zeros{T}((Base.OneTo(1),blockedrange(Fill(2,∞)))),
        BlockBroadcastVector{T}(vcat,
            unitblocks((b+2)^2 ./ (2*(2n .+ (b+2)) .* (2n .+ (b+4))) .+ one(T)/2),
            unitblocks(b^2 ./ (2*(2n .+ (b+2)) .* (2n .+ (b+4))) .+ one(T)/2))',
        Zeros{T}((Base.OneTo(1),blockedrange(Fill(2,∞)))),
        BlockBroadcastVector{T}(vcat,
            unitblocks((n .+ 1) .* (n .+ (b+3)) ./ ((2n .+ (b+3)) .* (2n .+ (b+4)))),
            unitblocks((n .+ 2) .* (n .+ (b+2)) ./ ((2n .+ (b+3)) .* (2n .+ (b+4)))))')), 2, 2)
end

function jacobimatrix(::Val{2}, P::VectorJacobi{T}) where T
    @assert P.𝐚 == (0,0)
    b = P.b
    k = 0:∞
    _vectorjacobibanded(
        BlockHcat(unitblocks([0,0,0,0, one(T),0,0]),
        BlockVcat(
            BlockBroadcastVector{T}(vcat,
                unitblocks(Zeros{T}(∞)),
                unitblocks(k .* (k .+ 1) ./ ((2k .+ (b+2)) .* (2k .+ (b+3)))))',
            Zeros{T}((Base.OneTo(1),blockedrange(Fill(2,∞)))),
            BlockBroadcastVector{T}(vcat,
                unitblocks((k .+ (b+1)) .* (k .+ (b+2)) ./ ((2k .+ (b+2)) .* (2k .+ (b+3)))),
                unitblocks(2 * (k .+ (b+2)) .* (k .+ 1) ./ ((2k .+ (b+2)) .* (2k .+ (b+4)))))',
            Zeros{T}((Base.OneTo(1),blockedrange(Fill(2,∞)))),
            BlockBroadcastVector{T}(vcat,
                unitblocks(2 * (k .+ (b+2)) .* (k .+ 1) ./ ((2k .+ (b+2)) .* (2k .+ (b+4)))),
                unitblocks((k .+ (b+2)) .* (k .+ (b+3)) ./ ((2k .+ (b+3)) .* (2k .+ (b+4)))))',
            Zeros{T}((Base.OneTo(1),blockedrange(Fill(2,∞)))),
            BlockBroadcastVector{T}(vcat,
            unitblocks((k .+ 2) .*  (k .+ 1) ./ ((2k .+ (b+3)) .* (2k .+ (b+4)))),
            unitblocks(Zeros{T}(∞)))'
        )), 3, 3)
end


####
# transforms
####

function transform_ldiv(Q::VectorJacobi{V}, f::AbstractQuasiVector) where V
    if Q.𝐚 == (0,0) && Q.b == 0
        t = axes(Q,1)
        c₊ = legendre(0..1) \ (sum.(f)/2)
        c₋ = jacobi(0,2,0..1) \ (only.(diff.(f)) ./ (-2t))
        n = max(last(colsupport(c₊)), last(colsupport(c₋)))
        pad(vec([c₊[1:n]'; c₋[1:n]']), axes(Q,2))
    else
        P = VectorJacobi{V}((0,0), 0)
        R = P \ Q
        R \ (P \ f)
    end
end


######
# ladder operators
######

const UnitJacobi{T} = SubQuasiArray{T, 2, Jacobi{T, Int}, Tuple{AffineMap{T, Inclusion{T, UnitInterval{T}}, Inclusion{T, ChebyshevInterval{T}}}, Base.Slice{OneToInf{Int}}}, false}

function laddergradmatrix(::Type{T}, a, b) where T
    n = zero(T):∞
    D = _BandedBlockBandedMatrix(
        BlockedMatrix(Vcat(Vcat(2b, -((n .+ 1) .* ((2a+b+2) .+ 2n) .* ((4+2a+b) .+ 2n)) ./ (2* ((a+b+2) .+ 2n) .* ((3+a+b) .+ 2n)))',
        (2 * (n .+ b))'), (OneTo(2), blockedrange(Ones{Int}(∞))))   ,
    BlockedOneTo(1:2:∞), (0,0), (1,0))
end

"""
    laddergrad(jacobi(a,b,0..1))

represents applying the vector-differential operator [1,0]*2t*d/dt + b*[1,1].
"""
function laddergrad(P::UnitJacobi{T}) where T
    (; a, b) = parent(P)
    VectorJacobi((a+1,a),b-1) * laddergradmatrix(T, a, b)
end

function laddercurlmatrix(::Type{T}, a, b) where T
    n = one(T):∞
    _BandedBlockBandedMatrix(
    BlockedArray(
            Vcat((4 * (n .+ (b+1)) .* (2n .+ (a+b+2)) .* (2n .+ (a+b+1)) .^2 ./ (𝔯.(n,a,b) .* n))',
                ((2n .+ (2a+b+1)) .* (2n .+ (2a+b+3)) .* (2n .+ (a+b+1)) ./ 𝔯.(n,a,b))'),
                (Base.OneTo(2), blockedrange(Ones{Int}(∞)))), 
                                BlockedOneTo(1:2:∞), (1,-1), (1,0))'
end

"""
    laddercurl(VectorJacobi((a+1,a), b))

represents applying the vector-differential operator [f_1,f_2] ↦ [1 0]*(b+1)/t*(f_2-f_1) + 2df_2/dt
"""
function laddercurl(P::VectorJacobi{T}) where T
    (a₁,a), b = P.𝐚,P.b
    @assert a₁ == a+1
    jacobi(a+1,b+1,UnitInterval{real(T)}()) * laddercurlmatrix(T, a, b)
end

function laddercurl_layout(::ApplyLayout{typeof(*)}, P)
    args = arguments(*, P)
    *(laddercurl(first(args)), tail(args)...)
end

laddercurl(P) = laddercurl_layout(MemoryLayout(P), P)

function laddergrad_layout(::ApplyLayout{typeof(*)}, P)
    args = arguments(*, P)
    *(laddergrad(first(args)), tail(args)...)
end
laddergrad(P) = laddergrad_layout(MemoryLayout(P), P)