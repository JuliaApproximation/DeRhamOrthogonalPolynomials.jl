module DeRhamExamples

using CairoMakie, LinearAlgebra
import CairoMakie.Makie

export symdiskplot, symdiskplot!, streamdiskplot, streamdiskplot!

indisk(x, y) = x^2 + y^2 ≤ 1
const unitcircle = Circle(Point2f(0, 0), 1f0)

"""
    streamdiskplot(f; kwargs...)

Stream plot of the vector field `f(x, y)` (returning 2 components) on the unit disk,
coloured by magnitude. Streamlines stop at the boundary.

    fig, ax, plt = streamdiskplot(f; axis=(; aspect=DataAspect()))
    Colorbar(fig[1, 2], plt, label="‖f‖")
"""
@recipe StreamDiskPlot (f,) begin
    "Colormap for the magnitude of the field."
    colormap = :viridis
    "Colour range for the magnitude; determined from the streamlines by default."
    colorrange = Makie.automatic
    "Density of streamlines."
    density = 1.0
    "Grid used to seed streamlines."
    gridsize = (32, 32)
    "Integration step of the streamlines."
    stepsize = 0.01
    "Maximum number of integration steps per streamline."
    maxsteps = 500
    arrow_size = 8
    linewidth = 1.5
    "Colour of the disk boundary."
    boundarycolor = :black
    boundarywidth = 1
end

function Makie.plot!(p::StreamDiskPlot)
    # NaN outside the disk terminates streamlines at the boundary
    map!(p.attributes, :f, :field) do f
        function (q)
            x, y = q
            indisk(x, y) || return Point2f(NaN, NaN)
            v = f(x, y)
            Point2f(v[1], v[2])
        end
    end

    streamplot!(p, p.field, -1 .. 1, -1 .. 1; p.colormap, p.colorrange, p.density, p.gridsize,
                p.stepsize, p.maxsteps, p.arrow_size, p.linewidth)
    lines!(p, unitcircle; color=p.boundarycolor, linewidth=p.boundarywidth)
    p
end

"""
    symdiskplot(f; kwargs...)

Plot the SPD matrix field `f(x, y)` on the unit disk as eigen-ellipses coloured by
anisotropy `(λ₂ - λ₁) / (λ₂ + λ₁)`, with principal directions, over a heatmap of the trace.

    fig, ax, plt = symdiskplot(f; axis=(; aspect=DataAspect()))
    Colorbar(fig[1, 2], plt, label="Anisotropy")            # eigen-ellipses
    Colorbar(fig[1, 3], plt.plots[1], label="Trace")        # background heatmap
"""
@recipe SymDiskPlot (f,) begin
    "Number of glyphs across the diameter."
    n = 11
    "Resolution of the background trace heatmap."
    nfine = 150
    "Glyph scale; `automatic` makes the largest glyph fill ~45% of the grid spacing."
    scale = Makie.automatic
    "Draw ellipses with radii `√λ` instead of `λ`."
    use_sqrt = false
    "Number of points per ellipse."
    npts = 48
    "Colormap for the anisotropy of the glyphs."
    colormap = :viridis
    colorrange = (0.0, 1.0)
    strokewidth = 1.0
    strokecolor = (:black, 0.7)
    "Colour and width of the major-axis segments."
    axiscolor = (:white, 0.85)
    axiswidth = 1.5
    "Colormap and transparency of the trace heatmap."
    tracecolormap = :greys
    tracealpha = 0.25
    "Colour of the disk boundary."
    boundarycolor = :black
    boundarywidth = 1
end

function Makie.plot!(p::SymDiskPlot)
    # Background: trace on a fine grid, NaN (transparent) outside the disk
    map!(p.attributes, [:f, :nfine], [:xs, :trace]) do f, nfine
        xs = range(-1, 1, length=nfine)
        xs, [indisk(x, y) ? Float64(tr(f(x, y))) : NaN for x in xs, y in xs]
    end

    map!(p.attributes, [:f, :n, :scale, :use_sqrt, :npts],
         [:ellipses, :anisotropies, :major_axes]) do f, n, scale, use_sqrt, npts
        # Glyph centres: coarse grid points inside the disk
        gs = range(-1, 1, length=n)
        centres = [(x, y) for x in gs, y in gs if indisk(x, y)]
        Ms = [Symmetric(f(x, y)) for (x, y) in centres]
        eigs = eigen.(Ms)

        # Glyph radius along the major axis (λ or √λ)
        radius(λ) = use_sqrt ? sqrt(λ) : λ
        if scale === Makie.automatic
            scale = 0.45 * step(gs) / maximum(radius(F.values[2]) for F in eigs)
        end

        ts = range(0, 2π, length=npts)
        ellipses = Vector{Point2f}[]
        anisotropies = Float64[]
        major_axes = Point2f[]

        for ((x, y), M, F) in zip(centres, Ms, eigs)
            λ1, λ2 = F.values
            v2 = Point2f(F.vectors[:, 2])
            T = use_sqrt ? sqrt(M) : M
            c = Point2f(x, y)

            push!(ellipses, [c + scale * Point2f(T * [cos(t), sin(t)]) for t in ts])
            push!(anisotropies, (λ2 - λ1) / (λ2 + λ1))
            r = scale * radius(λ2)
            push!(major_axes, c - r * v2, c + r * v2)
        end
        ellipses, anisotropies, major_axes
    end

    heatmap!(p, p.xs, p.xs, p.trace; colormap=p.tracecolormap, alpha=p.tracealpha)
    poly!(p, p.ellipses; color=p.anisotropies, p.colormap, p.colorrange,
          p.strokewidth, p.strokecolor)
    linesegments!(p, p.major_axes; color=p.axiscolor, linewidth=p.axiswidth)
    lines!(p, unitcircle; color=p.boundarycolor, linewidth=p.boundarywidth)
    p
end

# Functions are passed through to the recipes unconverted
Makie.convert_arguments(::Type{<:Union{StreamDiskPlot,SymDiskPlot}}, f::Function) = (f,)

# `Colorbar(pos, plt)` refers to the anisotropy of the glyphs (the heatmap is `plt.plots[1]`)
Makie.extract_colormap(p::SymDiskPlot) = Makie.extract_colormap_recursive(p.plots[2])

end # module DeRhamExamples
