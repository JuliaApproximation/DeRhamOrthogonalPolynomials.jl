
#######
# 2D
########

harmonicpolynomial(m, (x,y)::StaticVector{2}) = (x+sign(m)*im*y)^abs(m)
function vectorharmonicpolynomial(m, 𝐱::StaticVector{2})
    r,θ = norm(𝐱),atan(reverse(𝐱)...)
    r^(abs(m)-1) * exp(im*m*θ) * @SMatrix([cos(θ) -sin(θ); sin(θ) cos(θ)]) * SVector(1, im*(1-2signbit(m)))
end

#######
# 3D
########

function harmonicpolynomial(ℓ, m, 𝐱::StaticVector{3,T}) where T
    r = norm(𝐱)
    if iszero(r)
        convert(complex(T), iszero(ℓ))
    else
        sphericalharmonicy(ℓ, m,  𝐱/r) * r^ℓ
    end
end


sphericalharmonicy_θ(ℓ, m, θ, φ) = sphericalharmonicnormalization(eltype(θ), ℓ, m) * (m*cot(θ)*associatedlegendre(ℓ,m,cos(θ)) + (m < ℓ) * associatedlegendre(ℓ,m+1,cos(θ))) * exp(im*m*φ)
sphericalharmonicy_θ(ℓ, m, 𝐱::SphericalCoordinate) = sphericalharmonicy_θ(ℓ, m, 𝐱.θ, 𝐱.φ)
sphericalharmonicy_θ(ℓ, m, 𝐱::StaticVector{3}) = sphericalharmonicy_θ(ℓ, m, convert(SphericalCoordinate, 𝐱))

function vectorharmonicpolynomials(ℓ, m, 𝐱::StaticVector{3})
    # hack
    if iszero(𝐱)
        𝐱 += SVector(1E-15, 0, 0)
    end
    x,y,z = 𝐱
    (; r, φ, θ) = SphericalCoordinate(𝐱)
    ℓ*harmonicpolynomial(ℓ, m, 𝐱)/r * 𝐱/r + r^ℓ * sphericalharmonicy_θ(ℓ, m, 𝐱)/r * SVector(x*z,y*z,-(x^2+y^2))/(r*sqrt(x^2+y^2)) + im*m*harmonicpolynomial(ℓ, m, 𝐱)/(r*sin(θ)) * SVector(-y,x,0)/sqrt(x^2+y^2)
end


vectorharmonicpolynomialr(ℓ, m, 𝐱::StaticVector{3}) = 𝐱 * harmonicpolynomial(ℓ, m, 𝐱)
vectorharmonicpolynomialx(ℓ, m, 𝐱::StaticVector{3}) = 𝐱 × vectorharmonicpolynomials(ℓ, m, 𝐱)
function vectorharmonicpolynomialu(ℓ, m, 𝐱::StaticVector{3})
    r² = norm_sqr(𝐱)
    vectorharmonicpolynomialr(ℓ, m, 𝐱) - r²/(2ℓ+1) * vectorharmonicpolynomials(ℓ, m, 𝐱)
end