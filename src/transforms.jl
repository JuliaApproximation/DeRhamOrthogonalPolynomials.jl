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
