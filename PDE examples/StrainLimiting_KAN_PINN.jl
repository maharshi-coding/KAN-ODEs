using Random
using Statistics
using LinearAlgebra
using Flux
using Zygote
using Printf
using Plots

struct KANLayer{T1,T2,T3,T4,T5}
    coeff::T1
    lin::T2
    bias::T3
    centers::T4
    logwidth::T5
end

Flux.@layer KANLayer trainable=(coeff, lin, bias, logwidth)

function KANLayer(in_dim::Int, out_dim::Int, n_basis::Int; scale::Float32 = 0.1f0)
    coeff = scale .* randn(Float32, out_dim, in_dim, n_basis)
    lin = scale .* randn(Float32, out_dim, in_dim)
    bias = zeros(Float32, out_dim)
    centers = collect(Float32, range(0f0, 1f0; length=n_basis))
    logwidth = fill(log(0.15f0), n_basis)
    return KANLayer(coeff, lin, bias, centers, logwidth)
end

function (layer::KANLayer)(x::AbstractMatrix)
    in_dim, batch = size(x)
    out_dim = size(layer.lin, 1)
    n_basis = length(layer.centers)
    widths = exp.(layer.logwidth) .+ 1f-5

    lin_part = layer.lin * x
    basis_part = zeros(eltype(x), out_dim, batch)

    @inbounds for i in 1:in_dim
        xi = reshape(view(x, i, :), 1, batch)
        bi = exp.(-((xi .- reshape(layer.centers, n_basis, 1)) ./ reshape(widths, n_basis, 1)).^2)
        ci = reshape(view(layer.coeff, :, i, :), out_dim, n_basis)
        basis_part = basis_part + ci * bi
    end

    return lin_part + basis_part .+ reshape(layer.bias, out_dim, 1)
end

function build_kan_model(; hidden::Int = 32, n_basis::Int = 16)
    return Chain(
        KANLayer(2, hidden, n_basis),
        x -> tanh.(x),
        KANLayer(hidden, hidden, n_basis),
        x -> tanh.(x),
        KANLayer(hidden, 1, n_basis),
    )
end

Base.@kwdef struct MaterialParams
    μ::Float32 = 1.0f0
    β::Float32 = 5.0f0
    α::Float32 = 2.0f0
end

Base.@kwdef struct GeometryParams
    xmin::Float32 = 0.0f0
    xmax::Float32 = 1.0f0
    ymin::Float32 = 0.0f0
    ymax::Float32 = 1.0f0
    tip::Tuple{Float32,Float32} = (0.5f0, 0.5f0)
    notch_angle::Float32 = deg2rad(20.0f0)
    notch_length::Float32 = 0.35f0
    refine_half_width::Float32 = 0.10f0
end

Base.@kwdef struct BCParams
    tau1::Float32 = 1.0f0
    tau3_a::Float32 = 1.0f0
    tau3_b::Float32 = 0.0f0
    tau4_a::Float32 = -1.0f0
    tau4_b::Float32 = 1.0f0
end

Base.@kwdef struct TrainParams
    epochs::Int = 1500
    n_interior_uniform::Int = 256
    n_interior_refine::Int = 256
    n_boundary_each::Int = 128
    λ_bc::Float32 = 20.0f0
    λ_gauge::Float32 = 1.0f-3
    learning_rate::Float32 = 1.0f-3
    print_every::Int = 50
    seed::Int = 42
end

function phi_scalar(model, x::Real, y::Real)
    xy = Float32[x, y]
    return first(model(reshape(xy, 2, 1)))
end

function grad_phi(model, x::Real, y::Real)
    f(v) = first(model(reshape(v, 2, 1)))
    g = Zygote.gradient(f, Float32[x, y])[1]
    return g[1], g[2]
end

function flux_components(model, x::Real, y::Real, mat::MaterialParams)
    gx, gy = grad_phi(model, x, y)
    gnorm = sqrt(gx * gx + gy * gy + 1f-12)
    denom = 2f0 * mat.μ * (1f0 + mat.β * gnorm^mat.α)^(1f0 / mat.α)
    return gx / denom, gy / denom
end

function divergence_flux(model, x::Real, y::Real, mat::MaterialParams)
    qx_fun(xv) = first(flux_components(model, xv, y, mat))
    qy_fun(yv) = last(flux_components(model, x, yv, mat))
    dqxdx = Zygote.gradient(qx_fun, Float32(x))[1]
    dqydy = Zygote.gradient(qy_fun, Float32(y))[1]
    return dqxdx + dqydy
end

function sample_interior(geo::GeometryParams, n_uniform::Int, n_refine::Int)
    xu = geo.xmin .+ (geo.xmax - geo.xmin) .* rand(Float32, n_uniform)
    yu = geo.ymin .+ (geo.ymax - geo.ymin) .* rand(Float32, n_uniform)

    x0, y0 = geo.tip
    hr = geo.refine_half_width
    xr = clamp.(x0 .+ (2f0 .* rand(Float32, n_refine) .- 1f0) .* hr, geo.xmin, geo.xmax)
    yr = clamp.(y0 .+ (2f0 .* rand(Float32, n_refine) .- 1f0) .* hr, geo.ymin, geo.ymax)

    x = vcat(xu, xr)
    y = vcat(yu, yr)
    return hcat(x, y)
end

function notch_face_points_and_normals(geo::GeometryParams, n::Int)
    x0, y0 = geo.tip
    θ = geo.notch_angle
    L = geo.notch_length

    ang1 = π - θ / 2f0
    ang2 = π + θ / 2f0

    d1 = Float32[cos(ang1), sin(ang1)]
    d2 = Float32[cos(ang2), sin(ang2)]

    s = rand(Float32, n) .* L

    p1 = hcat(x0 .+ s .* d1[1], y0 .+ s .* d1[2])
    p2 = hcat(x0 .+ s .* d2[1], y0 .+ s .* d2[2])

    n1 = Float32[-d1[2], d1[1]]
    n2 = Float32[d2[2], -d2[1]]
    n1 ./= norm(n1)
    n2 ./= norm(n2)

    return p1, p2, n1, n2
end

function sample_boundaries(geo::GeometryParams, n_each::Int)
    x1 = geo.xmin .+ (geo.xmax - geo.xmin) .* rand(Float32, n_each)
    Γ1 = hcat(x1, fill(geo.ymax, n_each))
    nΓ1 = fill((0f0, 1f0), n_each)

    x2 = geo.xmin .+ (geo.xmax - geo.xmin) .* rand(Float32, n_each)
    Γ2 = hcat(x2, fill(geo.ymin, n_each))
    nΓ2 = fill((0f0, -1f0), n_each)

    y3 = geo.ymin .+ (geo.ymax - geo.ymin) .* rand(Float32, n_each)
    Γ3 = hcat(fill(geo.xmin, n_each), y3)
    nΓ3 = fill((-1f0, 0f0), n_each)

    y4 = geo.ymin .+ (geo.ymax - geo.ymin) .* rand(Float32, n_each)
    Γ4 = hcat(fill(geo.xmax, n_each), y4)
    nΓ4 = fill((1f0, 0f0), n_each)

    p5a, p5b, n5a, n5b = notch_face_points_and_normals(geo, n_each)
    nΓ5a = fill((n5a[1], n5a[2]), n_each)
    nΓ5b = fill((n5b[1], n5b[2]), n_each)

    return (
        Γ1 = (pts = Γ1, normals = nΓ1),
        Γ2 = (pts = Γ2, normals = nΓ2),
        Γ3 = (pts = Γ3, normals = nΓ3),
        Γ4 = (pts = Γ4, normals = nΓ4),
        Γ5a = (pts = p5a, normals = nΓ5a),
        Γ5b = (pts = p5b, normals = nΓ5b),
    )
end

function traction_target(label::Symbol, x::Float32, y::Float32, bc::BCParams)
    if label === :Γ1
        return bc.tau1
    elseif label === :Γ2
        return 0f0
    elseif label === :Γ3
        return bc.tau3_a * y + bc.tau3_b
    elseif label === :Γ4
        return bc.tau4_a * y + bc.tau4_b
    else
        return 0f0
    end
end

function pde_loss(model, interior_pts::Matrix{Float32}, mat::MaterialParams)
    vals = map(eachrow(interior_pts)) do p
        r = divergence_flux(model, p[1], p[2], mat)
        r * r
    end
    return mean(vals)
end

function boundary_loss(model, bdata, mat::MaterialParams, bc::BCParams)
    labels = (:Γ1, :Γ2, :Γ3, :Γ4, :Γ5a, :Γ5b)
    losses = Float32[]

    for lbl in labels
        pts = getfield(bdata, lbl).pts
        normals = getfield(bdata, lbl).normals
        for i in 1:size(pts, 1)
            x = pts[i, 1]
            y = pts[i, 2]
            nx, ny = normals[i]
            qx, qy = flux_components(model, x, y, mat)
            tr = qx * nx + qy * ny
            tgt = traction_target(lbl, x, y, bc)
            push!(losses, (tr - tgt)^2)
        end
    end

    return mean(losses)
end

function gauge_loss(model)
    return phi_scalar(model, 0f0, 0f0)^2
end

function train!(θ, re, mat, geo, bc, trn)
    opt = ADAM(trn.learning_rate)

    total_hist = Float32[]
    pde_hist = Float32[]
    bc_hist = Float32[]
    t0 = time()

    for epoch in 1:trn.epochs
        interior = sample_interior(geo, trn.n_interior_uniform, trn.n_interior_refine)
        bdata = sample_boundaries(geo, trn.n_boundary_each)

        function loss_fn(θvec)
            model = re(θvec)
            lpde = pde_loss(model, interior, mat)
            lbc = boundary_loss(model, bdata, mat, bc)
            lg = gauge_loss(model)
            return lpde + trn.λ_bc * lbc + trn.λ_gauge * lg, lpde, lbc
        end

        (lval, lpde, lbc), back = Zygote.pullback(loss_fn, θ)
        g = first(back((1f0, 0f0, 0f0)))
        Flux.update!(opt, θ, g)

        push!(total_hist, lval)
        push!(pde_hist, lpde)
        push!(bc_hist, lbc)

        if epoch == 1 || epoch % trn.print_every == 0
            elapsed = time() - t0
            sec_per_epoch = elapsed / epoch
            remaining = sec_per_epoch * (trn.epochs - epoch)
            @printf("Epoch %5d/%d | L=%.5e | Lpde=%.5e | Lbc=%.5e | %.2fs/ep | ETA %.1f min\n",
                epoch, trn.epochs, lval, lpde, lbc, sec_per_epoch, remaining / 60)
        end
    end

    return θ, total_hist, pde_hist, bc_hist
end

function field_on_grid(model, geo::GeometryParams; nx::Int = 121, ny::Int = 121)
    xs = collect(range(geo.xmin, geo.xmax; length=nx))
    ys = collect(range(geo.ymin, geo.ymax; length=ny))
    Φ = [phi_scalar(model, x, y) for y in ys, x in xs]
    return xs, ys, Φ
end

function gradmag_on_line(model, xline::AbstractVector{<:Real}, yline::AbstractVector{<:Real})
    return [begin gx, gy = grad_phi(model, x, y); sqrt(gx^2 + gy^2) end for (x, y) in zip(xline, yline)]
end

function save_plots(model, loss_hist, pde_hist, bc_hist, geo::GeometryParams; outdir::String = "results_strainlimiting")
    mkpath(outdir)

    p1 = plot(loss_hist; yscale=:log10, lw=2, label="L total", xlabel="Epoch", ylabel="Loss", title="Training history")
    plot!(p1, pde_hist; lw=2, label="L_pde")
    plot!(p1, bc_hist; lw=2, label="L_bc")
    savefig(p1, joinpath(outdir, "loss_history.png"))

    xs, ys, Φ = field_on_grid(model, geo)
    p2 = heatmap(xs, ys, Φ; color=:turbo, xlabel="x", ylabel="y", title="Stress function Φ(x,y)")
    savefig(p2, joinpath(outdir, "phi_field.png"))

    x0, y0 = geo.tip
    xline = collect(range(geo.xmin, x0; length=250))
    yline = fill(y0, length(xline))
    gline = gradmag_on_line(model, xline, yline)
    dist_to_tip = x0 .- xline

    p3 = plot(dist_to_tip, gline; lw=2, xlabel="Distance to notch tip", ylabel="|∇Φ|", title="Gradient magnitude along reference line", label="|∇Φ|")
    savefig(p3, joinpath(outdir, "gradmag_reference_line.png"))
end

function main()
    mat = MaterialParams()
    geo = GeometryParams()
    bc = BCParams()
    trn = TrainParams(
        epochs=parse(Int, get(ENV, "KAN_PINN_EPOCHS", "1500")),
        n_interior_uniform=parse(Int, get(ENV, "KAN_PINN_NU", "256")),
        n_interior_refine=parse(Int, get(ENV, "KAN_PINN_NR", "256")),
        n_boundary_each=parse(Int, get(ENV, "KAN_PINN_NB", "128")),
        learning_rate=parse(Float32, get(ENV, "KAN_PINN_LR", "1.0e-3")),
        print_every=parse(Int, get(ENV, "KAN_PINN_PRINT_EVERY", "50")),
        seed=parse(Int, get(ENV, "KAN_PINN_SEED", "42")),
    )

    Random.seed!(trn.seed)

    model0 = build_kan_model(hidden=32, n_basis=16)
    θ0, re = Flux.destructure(model0)
    θ = copy(θ0)

    θ, lhist, lpde_hist, lbc_hist = train!(θ, re, mat, geo, bc, trn)
    model = re(θ)

    save_plots(model, lhist, lpde_hist, lbc_hist, geo; outdir=joinpath(@__DIR__, "results_strainlimiting"))
    println("Training complete. Outputs saved in: ", joinpath(@__DIR__, "results_strainlimiting"))
end

main()
