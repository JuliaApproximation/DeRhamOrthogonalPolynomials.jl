module DeRhamExamples

using CairoMakie, LinearAlgebra

export symdiskplot, streamdiskplot

"""
    streamdiskplot(f; nfine=150, density=1.0, colormap=:viridis, title=...)

Stream plot of the vector field `f(x, y)` (returning 2 components) on the unit disk,
coloured by magnitude, over a heatmap of the magnitude. Streamlines stop at the boundary.
"""
function streamdiskplot(f; nfine=150, density=1.0, colormap=:viridis, gridsize=(32, 32),
                           stepsize=0.01, maxsteps=500, title="Vector Field on the Unit Disk")
    indisk(x, y) = x^2 + y^2 ≤ 1
    vec(x, y) = (v = f(x, y); Point2f(v[1], v[2]))

    # Background: magnitude on a fine grid, NaN (transparent) outside the disk
    xs = ys = range(-1, 1, length=nfine)
    mag_grid = [indisk(x, y) ? Float64(norm(vec(x, y))) : NaN for x in xs, y in ys]
    crange = extrema(filter(!isnan, mag_grid))
    crange[1] == crange[2] && (crange = (crange[1], crange[1] + 1))

    # NaN outside the disk terminates streamlines at the boundary
    field(p) = indisk(p[1], p[2]) ? vec(p[1], p[2]) : Point2f(NaN, NaN)

    fig = Figure(size=(760, 680), fontsize=14)
    ax = Axis(fig[1, 1]; title, xlabel="x", ylabel="y", aspect=DataAspect())

    heatmap!(ax, xs, ys, mag_grid; colormap=:greys, alpha=0.25)
    streamplot!(ax, field, -1..1, -1..1; colormap, colorrange=crange, density,
                 gridsize, stepsize, maxsteps, arrow_size=8, linewidth=1.5)
    lines!(ax, Circle(Point2f(0, 0), 1f0); color=:black, linewidth=1)  # disk boundary
    limits!(ax, -1.05, 1.05, -1.05, 1.05)

    Colorbar(fig[1, 2]; colormap, colorrange=crange, label="Magnitude  ‖f‖")

    fig
end

"""
    symdiskplot(f; n=11, nfine=150, scale=nothing, use_sqrt=false, npts=48)

Plot the SPD matrix field `f(x, y)` on the unit disk as eigen-ellipses coloured by
anisotropy, over a heatmap of the trace. `scale=nothing` sizes glyphs automatically.
"""
function symdiskplot(f; n=11, nfine=150, scale=nothing, use_sqrt=false, npts=48,
                        title="2D SPD Matrix Field (Eigen-Ellipses & Principal Directions)")
    indisk(x, y) = x^2 + y^2 ≤ 1

    # Background: trace on a fine grid, NaN (transparent) outside the disk
    xs = ys = range(-1, 1, length=nfine)
    trace_grid = [indisk(x, y) ? tr(f(x, y)) : NaN for x in xs, y in ys]

    # Glyph centres: coarse grid points inside the disk
    gs = range(-1, 1, length=n)
    centres = [(x, y) for x in gs, y in gs if indisk(x, y)]
    eigs = [eigen(Symmetric(f(x, y))) for (x, y) in centres]

    # Glyph radius along the major axis (λ or √λ)
    radius(λ) = use_sqrt ? sqrt(λ) : λ
    if scale === nothing
        # largest glyph fills ~45% of the grid spacing
        scale = 0.45 * step(gs) / maximum(radius(F.values[2]) for F in eigs)
    end

    ts = range(0, 2π, length=npts)
    ellipses = Vector{Point2f}[]
    anisotropies = Float64[]
    major_axes = Tuple{Point2f,Point2f}[]

    for ((x, y), F) in zip(centres, eigs)
        λ1, λ2 = F.values
        v2 = F.vectors[:, 2]
        M = Symmetric(f(x, y))
        T = use_sqrt ? sqrt(M) : M
        c = Point2f(x, y)

        push!(ellipses, [c + scale * Point2f(T * [cos(t), sin(t)]) for t in ts])
        push!(anisotropies, (λ2 - λ1) / (λ2 + λ1))
        r = scale * radius(λ2)
        push!(major_axes, (c - r * Point2f(v2), c + r * Point2f(v2)))
    end

    fig = Figure(size=(850, 680), fontsize=14)
    ax = Axis(fig[1, 1]; title, xlabel="x", ylabel="y", aspect=DataAspect())

    hm = heatmap!(ax, xs, ys, trace_grid, colormap=:greys, alpha=0.25)
    plt = poly!(ax, ellipses; color=anisotropies, colormap=:viridis, colorrange=(0.0, 1.0),
                strokewidth=1.0, strokecolor=(:black, 0.7))
    linesegments!(ax, major_axes; color=(:white, 0.85), linewidth=1.5)
    lines!(ax, Circle(Point2f(0, 0), 1f0); color=:black, linewidth=1)  # disk boundary

    Colorbar(fig[1, 2], plt, label="Anisotropy  (λ₂ - λ₁) / (λ₂ + λ₁)")
    Colorbar(fig[1, 3], hm, label="Trace  tr(M) = λ₁ + λ₂")

    fig
end



end # module DeRhamExamples
