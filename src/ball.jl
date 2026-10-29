const BallBubbleWeight{T} = HyperZernikeWeight{3,T}
const BallVectorWeight{T} = HyperBallVectorWeight{3,T,9}
BallVectorWeight() = BallVectorWeight{Float64}()


const AbstractBallBasis{T} = AbstractHyperZernike{3,T}


"""
    L^2 basis
"""
struct BallBasis{T} <: AbstractBallBasis{T}
    ℓ::Int
    m::Int
end

BallBasis(ℓ, m) = BallBasis{Float64}(ℓ, m)


==(A::BallBasis, B::BallBasis) = A.ℓ == B.ℓ && A.m == B.m

function zernike(ℓ, m, j::Int, λ, 𝐱::StaticVector{3,T}) where T
    r² = LinearAlgebra.norm_sqr(𝐱)
    harmonicpolynomial(ℓ, m, 𝐱) * jacobip(j, λ, ℓ+one(real(T))/2, 2r²-1)
end

Base.unsafe_getindex(P::BallBasis{T}, 𝐱::StaticVector{3}, j::Int) where T = zernike(P.ℓ, P.m, j-1, 0, 𝐱)

struct BallBubble{T} <: AbstractBallBasis{T}
    ℓ::Int
    m::Int
end

BallBubble(m) = BallBubble{Float64}(m)


==(A::BallBubble, B::BallBubble) = A.m == B.m && A.ℓ == B.ℓ
==(::BallBubble, ::BallBasis) = false
==(::BallBasis, ::BallBubble) = false

Base.unsafe_getindex(P::BallBubble, 𝐱::StaticVector{3}, j::Int) = zernike(P.ℓ, P.m, j-1, 1, 𝐱)
orthogonalityweight(::BallBubble{T}) where {T} = BallBubbleWeight{T}()

function vectorzernike(::typeof(+), ℓ, m, j, 𝐱::StaticVector{3})
    r² = LinearAlgebra.norm_sqr(𝐱) # = s
    jacobip(j, 0, ℓ+3/2, 2r²-1) * vectorharmonicpolynomialu(ℓ, m, 𝐱)
end

function vectorzernike(::typeof(-), ℓ, m, j, 𝐱::StaticVector{3})
    r² = LinearAlgebra.norm_sqr(𝐱) # = s
    jacobip(j, 0, ℓ-1/2, 2r²-1) * vectorharmonicpolynomials(ℓ, m, 𝐱)
end

function vectorzernike(::typeof(×), ℓ, m, j, 𝐱::StaticVector{3})
    r² = LinearAlgebra.norm_sqr(𝐱) # = s
    jacobip(j, 0, ℓ+1/2, 2r²-1) * vectorharmonicpolynomialx(ℓ, m, 𝐱)
end


function zernikegrad(ℓ, m, j::Int, λ, 𝐱::StaticVector{3,T}) where T
    m == ℓ == j == 0 && return zero(complex(T))*𝐱
    r² = LinearAlgebra.norm_sqr(𝐱)
    vectorharmonicpolynomials(ℓ, m, 𝐱) * jacobip(j, λ, ℓ+one(real(T))/2, 2r²-1) + (iszero(j) ? 0 : 2 * (j+λ+ℓ+T(3)/2) * jacobip(j-1, λ+1, ℓ+real(T(3))/2, 2r²-1) * harmonicpolynomial(ℓ, m, 𝐱)) * 𝐱
end