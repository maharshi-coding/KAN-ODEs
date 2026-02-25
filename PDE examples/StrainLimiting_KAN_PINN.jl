using Random
using Statistics
using LinearAlgebra
using Flux
using Zygote
using Printf
using Serialization
using Dates
ENV["GKSwstype"] = "100"
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
    notch_length::Float32 = 0.50f0
    refine_half_width::Float32 = 0.10f0
end

Base.@kwdef struct BCParams
    σ0::Float32 = 1.0f0
    L::Float32 = 1.0f0
end

Base.@kwdef struct TrainParams
    adam_epochs::Int = 1200
    finetune_epochs::Int = 300
    n_interior_uniform::Int = 96
    n_interior_refine::Int = 96
    n_interior_tip_strip::Int = 128
    n_boundary_each::Int = 48
    val_n_interior_uniform::Int = 192
    val_n_interior_refine::Int = 192
    val_n_interior_tip_strip::Int = 256
    val_n_boundary_each::Int = 96
    λ_bc::Float32 = 20.0f0
    λ_gauge::Float32 = 1.0f-3
    learning_rate::Float32 = 1.0f-3
    finetune_lr::Float32 = 2.0f-4
    print_every::Int = 50
    fd_eps::Float32 = 2.0f-3
    gc_every::Int = 25
    early_stop_patience::Int = 300
    min_improve::Float32 = 1.0f-4
    max_grad_norm::Float32 = 10.0f0
    checkpoint_every::Int = 100
    tip_strip_half_height::Float32 = 0.035f0
    tip_strip_length::Float32 = 0.20f0
    seed::Int = 42
end

function phi_scalar(model, x::Real, y::Real)
    xy = Float32[x, y]
    return first(model(reshape(xy, 2, 1)))
end

function finite_difference_1d(f, a::Float32, b::Float32, x::Float32, h::Float32)
    xph = min(x + h, b)
    xmh = max(x - h, a)
    if xph == xmh
        return 0f0
    elseif xph == x
        return (f(x) - f(xmh)) / (x - xmh + 1f-12)
    elseif xmh == x
        return (f(xph) - f(x)) / (xph - x + 1f-12)
    else
        return (f(xph) - f(xmh)) / (xph - xmh)
    end
end

function grad_phi_fd(model, x::Float32, y::Float32, geo::GeometryParams, h::Float32)
    fx(xv) = phi_scalar(model, xv, y)
    fy(yv) = phi_scalar(model, x, yv)
    gx = finite_difference_1d(fx, geo.xmin, geo.xmax, x, h)
    gy = finite_difference_1d(fy, geo.ymin, geo.ymax, y, h)
    return gx, gy
end

function flux_components(model, x::Float32, y::Float32, mat::MaterialParams, geo::GeometryParams, h::Float32)
    gx, gy = grad_phi_fd(model, x, y, geo, h)
    gnorm = sqrt(gx * gx + gy * gy + 1f-12)
    denom = 2f0 * mat.μ * (1f0 + mat.β * gnorm^mat.α)^(1f0 / mat.α)
    return gx / denom, gy / denom
end

function divergence_flux_fd(model, x::Float32, y::Float32, mat::MaterialParams, geo::GeometryParams, h::Float32)
    qx_fun(xv) = first(flux_components(model, xv, y, mat, geo, h))
    qy_fun(yv) = last(flux_components(model, x, yv, mat, geo, h))
    dqxdx = finite_difference_1d(qx_fun, geo.xmin, geo.xmax, x, h)
    dqydy = finite_difference_1d(qy_fun, geo.ymin, geo.ymax, y, h)
    return dqxdx + dqydy
end

function notch_face_directions(geo::GeometryParams)
    θ = geo.notch_angle
    d_upper = Float32[cos(θ / 2f0), sin(θ / 2f0)]
    d_lower = Float32[cos(θ / 2f0), -sin(θ / 2f0)]
    return d_upper, d_lower
end

function notch_mouth_points(geo::GeometryParams)
    x0, y0 = geo.tip
    d_upper, d_lower = notch_face_directions(geo)
    pu = Float32[x0 + geo.notch_length * d_upper[1], y0 + geo.notch_length * d_upper[2]]
    pl = Float32[x0 + geo.notch_length * d_lower[1], y0 + geo.notch_length * d_lower[2]]
    return pu, pl
end

function point_in_notch_void(x::Float32, y::Float32, geo::GeometryParams)
    x0, y0 = geo.tip
    if x < x0
        return false
    end
    dx = x - x0
    if dx > geo.notch_length
        return false
    end
    half_open = tan(geo.notch_angle / 2f0) * dx
    return abs(y - y0) <= half_open
end

function sample_points_excluding_notch(geo::GeometryParams, n::Int; xlo::Float32 = geo.xmin, xhi::Float32 = geo.xmax, ylo::Float32 = geo.ymin, yhi::Float32 = geo.ymax)
    pts = Matrix{Float32}(undef, n, 2)
    k = 1
    while k <= n
        x = xlo + (xhi - xlo) * rand(Float32)
        y = ylo + (yhi - ylo) * rand(Float32)
        if !point_in_notch_void(x, y, geo)
            pts[k, 1] = x
            pts[k, 2] = y
            k += 1
        end
    end
    return pts
end

function sample_interior(
    geo::GeometryParams,
    n_uniform::Int,
    n_refine::Int,
    n_tip_strip::Int,
    tip_strip_half_height::Float32,
    tip_strip_length::Float32,
)
    uniform_pts = sample_points_excluding_notch(geo, n_uniform)
    x0, y0 = geo.tip
    hr = geo.refine_half_width
    refine_pts = sample_points_excluding_notch(
        geo,
        n_refine;
        xlo=max(geo.xmin, x0 - hr),
        xhi=min(geo.xmax, x0 + hr),
        ylo=max(geo.ymin, y0 - hr),
        yhi=min(geo.ymax, y0 + hr),
    )

    tip_strip_pts = sample_points_excluding_notch(
        geo,
        n_tip_strip;
        xlo=max(geo.xmin, x0 - tip_strip_length),
        xhi=min(geo.xmax, x0),
        ylo=max(geo.ymin, y0 - tip_strip_half_height),
        yhi=min(geo.ymax, y0 + tip_strip_half_height),
    )

    return vcat(uniform_pts, refine_pts, tip_strip_pts)
end

function notch_face_points(geo::GeometryParams, n::Int)
    x0, y0 = geo.tip
    d1, d2 = notch_face_directions(geo)

    s = rand(Float32, n) .* geo.notch_length

    p1 = hcat(x0 .+ s .* d1[1], y0 .+ s .* d1[2])
    p2 = hcat(x0 .+ s .* d2[1], y0 .+ s .* d2[2])

    return p1, p2
end

function sample_boundaries(geo::GeometryParams, n_each::Int)
    y1 = geo.ymin .+ (geo.ymax - geo.ymin) .* rand(Float32, n_each)
    Γ1 = hcat(fill(geo.xmin, n_each), y1)

    x3 = geo.xmin .+ (geo.xmax - geo.xmin) .* rand(Float32, n_each)
    Γ3 = hcat(x3, fill(geo.ymin, n_each))

    x4 = geo.xmin .+ (geo.xmax - geo.xmin) .* rand(Float32, n_each)
    Γ4 = hcat(x4, fill(geo.ymax, n_each))

    pu, pl = notch_mouth_points(geo)
    ylo = clamp(min(pl[2], pu[2]), geo.ymin, geo.ymax)
    yhi = clamp(max(pl[2], pu[2]), geo.ymin, geo.ymax)
    right_pts = Matrix{Float32}(undef, n_each, 2)
    for i in 1:n_each
        y = geo.ymin + (geo.ymax - geo.ymin) * rand(Float32)
        while y >= ylo && y <= yhi
            y = geo.ymin + (geo.ymax - geo.ymin) * rand(Float32)
        end
        right_pts[i, 1] = geo.xmax
        right_pts[i, 2] = y
    end
    Γ2 = right_pts

    p5a, p5b = notch_face_points(geo, n_each)

    return (
        Γ1 = (pts = Γ1,),
        Γ2 = (pts = Γ2,),
        Γ3 = (pts = Γ3,),
        Γ4 = (pts = Γ4,),
        Γ5a = (pts = p5a,),
        Γ5b = (pts = p5b,),
    )
end

function dirichlet_target(label::Symbol, x::Float32, y::Float32, bc::BCParams)
    if label === :Γ1
        return bc.σ0 * bc.L
    elseif label === :Γ2
        return 0f0
    elseif label === :Γ3
        return -bc.σ0 * (x - bc.L)
    elseif label === :Γ4
        return bc.σ0 * (bc.L - x)
    else
        return 0f0
    end
end

function pde_loss(model, interior_pts::Matrix{Float32}, mat::MaterialParams, geo::GeometryParams, h::Float32)
    vals = map(eachrow(interior_pts)) do p
        r = divergence_flux_fd(model, p[1], p[2], mat, geo, h)
        r * r
    end
    return mean(vals)
end

function boundary_loss(model, bdata, bc::BCParams)
    labels = (:Γ1, :Γ2, :Γ3, :Γ4)
    total = 0f0
    count = 0

    for lbl in labels
        pts = getfield(bdata, lbl).pts
        for i in 1:size(pts, 1)
            x = pts[i, 1]
            y = pts[i, 2]
            ϕ = phi_scalar(model, x, y)
            g = dirichlet_target(lbl, x, y, bc)
            total += (ϕ - g)^2
            count += 1
        end
    end

    return total / max(count, 1)
end

function residual_statistics(model, mat::MaterialParams, geo::GeometryParams, trn::TrainParams; n::Int = 512)
    pts = sample_points_excluding_notch(geo, n)
    residuals = Float32[]
    for i in 1:size(pts, 1)
        x = pts[i, 1]
        y = pts[i, 2]
        r = divergence_flux_fd(model, x, y, mat, geo, trn.fd_eps)
        push!(residuals, r)
    end

    abs_res = abs.(residuals)
    rms = sqrt(mean(residuals .^ 2))
    return (
        mean_abs = mean(abs_res),
        max_abs = maximum(abs_res),
        rms = rms,
    )
end

function symmetry_error(model, geo::GeometryParams; n::Int = 512)
    x0, y0 = geo.tip
    _ = x0
    pts = sample_points_excluding_notch(geo, n; ylo=y0, yhi=geo.ymax)
    errs = Float32[]

    for i in 1:size(pts, 1)
        x = pts[i, 1]
        y = pts[i, 2]
        ym = Float32(2f0 * y0 - y)
        if ym >= geo.ymin && ym <= geo.ymax && !point_in_notch_void(x, ym, geo)
            ϕ1 = phi_scalar(model, x, y)
            ϕ2 = phi_scalar(model, x, ym)
            push!(errs, abs(ϕ1 - ϕ2))
        end
    end

    if isempty(errs)
        return (mean_abs = NaN32, max_abs = NaN32, n_pairs = 0)
    end

    return (mean_abs = mean(errs), max_abs = maximum(errs), n_pairs = length(errs))
end

function tip_gradient_indicator(model, geo::GeometryParams, trn::TrainParams)
    x0, y0 = geo.tip
    xnear = collect(range(max(geo.xmin, x0 - 0.06f0), x0 - 0.005f0; length=80))
    xfar = collect(range(geo.xmin, max(geo.xmin, x0 - 0.20f0); length=80))
    ynear = fill(y0, length(xnear))
    yfar = fill(y0, length(xfar))

    gnear = gradmag_on_line(model, xnear, ynear, geo; h=trn.fd_eps)
    gfar = gradmag_on_line(model, xfar, yfar, geo; h=trn.fd_eps)
    near_mean = mean(gnear)
    far_mean = mean(gfar)
    ratio = near_mean / (far_mean + 1f-8)

    return (near_mean = near_mean, far_mean = far_mean, ratio = ratio)
end

function grid_finite_check(model, geo::GeometryParams; nx::Int = 121, ny::Int = 121)
    xs, ys, Φ = field_on_grid(model, geo; nx=nx, ny=ny)
    bad_outside = 0
    outside_total = 0

    for (iy, y) in pairs(ys), (ix, x) in pairs(xs)
        inside_void = point_in_notch_void(Float32(x), Float32(y), geo)
        v = Φ[iy, ix]
        if !inside_void
            outside_total += 1
            if isnan(v) || !isfinite(v)
                bad_outside += 1
            end
        end
    end

    return (outside_total = outside_total, bad_outside = bad_outside)
end

function run_cross_verification(model, mat::MaterialParams, geo::GeometryParams, trn::TrainParams)
    rstats = residual_statistics(model, mat, geo, trn)
    sstats = symmetry_error(model, geo)
    tipstats = tip_gradient_indicator(model, geo, trn)
    gstats = grid_finite_check(model, geo)

    println("Cross verification summary:")
    @printf("  PDE residual  | mean|r|=%.5e, rms=%.5e, max|r|=%.5e\n", rstats.mean_abs, rstats.rms, rstats.max_abs)
    @printf("  Symmetry      | mean|ΔΦ|=%.5e, max|ΔΦ|=%.5e (pairs=%d)\n", sstats.mean_abs, sstats.max_abs, sstats.n_pairs)
    @printf("  Tip gradient  | near=%.5e, far=%.5e, near/far=%.3f\n", tipstats.near_mean, tipstats.far_mean, tipstats.ratio)
    @printf("  Finite check  | bad outside notch=%d / %d\n", gstats.bad_outside, gstats.outside_total)

    return (residual=rstats, symmetry=sstats, tip=tipstats, finite=gstats)
end

function gauge_loss(model)
    return phi_scalar(model, 0f0, 0f0)^2
end

function compute_losses(model, interior, bdata, mat, geo, bc, trn)
    lpde = pde_loss(model, interior, mat, geo, trn.fd_eps)
    lbc = boundary_loss(model, bdata, bc)
    lg = gauge_loss(model)
    ltot = lpde + trn.λ_bc * lbc + trn.λ_gauge * lg
    return ltot, lpde, lbc
end

function clip_gradient(g::AbstractVector{<:Real}, max_norm::Float32)
    if max_norm <= 0f0
        return g
    end
    gnorm = norm(g)
    if gnorm <= max_norm || gnorm == 0
        return g
    end
    scale = max_norm / Float32(gnorm)
    return g .* scale
end

function save_checkpoint(
    path::String,
    θ,
    best_θ,
    total_hist,
    pde_hist,
    bc_hist,
    val_hist,
    best_epoch,
    best_val,
    current_epoch::Int,
    val_interior,
    val_bdata,
)
    state = Dict(
        "theta" => copy(θ),
        "best_theta" => copy(best_θ),
        "loss_total" => copy(total_hist),
        "loss_pde" => copy(pde_hist),
        "loss_bc" => copy(bc_hist),
        "loss_val" => copy(val_hist),
        "best_epoch" => best_epoch,
        "best_val" => best_val,
        "current_epoch" => current_epoch,
        "val_interior" => val_interior,
        "val_bdata" => val_bdata,
    )
    serialize(path, state)
end

function train_stage!(
    θ,
    re,
    mat,
    geo,
    bc,
    trn;
    epochs::Int,
    lr::Float32,
    start_epoch::Int,
    best_θ,
    best_val::Float32,
    best_epoch::Int,
    stale_epochs::Int,
    total_hist,
    pde_hist,
    bc_hist,
    val_hist,
    val_interior,
    val_bdata,
    t0,
    ckpt_path::String,
)
    opt = Flux.Adam(lr)
    opt_state = Flux.setup(opt, θ)

    for stage_step in 1:epochs
        epoch = start_epoch + stage_step - 1

        interior = sample_interior(
            geo,
            trn.n_interior_uniform,
            trn.n_interior_refine,
            trn.n_interior_tip_strip,
            trn.tip_strip_half_height,
            trn.tip_strip_length,
        )
        bdata = sample_boundaries(geo, trn.n_boundary_each)

        function loss_fn(θvec)
            model = re(θvec)
            return compute_losses(model, interior, bdata, mat, geo, bc, trn)
        end

        (lval, lpde, lbc), back = Zygote.pullback(loss_fn, θ)
        g = first(back((1f0, 0f0, 0f0)))
        g = clip_gradient(g, trn.max_grad_norm)
        Flux.update!(opt_state, θ, g)

        push!(total_hist, lval)
        push!(pde_hist, lpde)
        push!(bc_hist, lbc)

        model_val = re(θ)
        lval_fixed, _, _ = compute_losses(model_val, val_interior, val_bdata, mat, geo, bc, trn)
        push!(val_hist, lval_fixed)

        if lval_fixed < best_val - trn.min_improve
            best_val = lval_fixed
            best_θ .= θ
            best_epoch = epoch
            stale_epochs = 0
        else
            stale_epochs += 1
        end

        if epoch == 1 || epoch % trn.print_every == 0
            elapsed = time() - t0
            sec_per_epoch = elapsed / epoch
            total_epochs = trn.adam_epochs + trn.finetune_epochs
            remaining = sec_per_epoch * (total_epochs - epoch)
            @printf("Epoch %5d/%d | L=%.5e | Lpde=%.5e | Lbc=%.5e | Lval=%.5e | %.2fs/ep | ETA %.1f min\n",
                epoch, total_epochs, lval, lpde, lbc, lval_fixed, sec_per_epoch, remaining / 60)
        end

        if trn.checkpoint_every > 0 && (epoch % trn.checkpoint_every == 0)
            save_checkpoint(
                ckpt_path,
                θ,
                best_θ,
                total_hist,
                pde_hist,
                bc_hist,
                val_hist,
                best_epoch,
                best_val,
                epoch,
                val_interior,
                val_bdata,
            )
        end

        if trn.gc_every > 0 && (epoch % trn.gc_every == 0)
            GC.gc(false)
        end

        if trn.early_stop_patience > 0 && stale_epochs >= trn.early_stop_patience
            println("Early stopping triggered at epoch ", epoch, ", best epoch = ", best_epoch)
            break
        end
    end

    return θ, best_θ, best_val, best_epoch, stale_epochs
end

function train!(θ, re, mat, geo, bc, trn; outdir::String, resume::Bool = false)
    ckpt_path = joinpath(outdir, "checkpoint_latest.jls")
    target_epochs = trn.adam_epochs + trn.finetune_epochs

    total_hist = Float32[]
    pde_hist = Float32[]
    bc_hist = Float32[]
    val_hist = Float32[]
    val_interior = sample_interior(
        geo,
        trn.val_n_interior_uniform,
        trn.val_n_interior_refine,
        trn.val_n_interior_tip_strip,
        trn.tip_strip_half_height,
        trn.tip_strip_length,
    )
    val_bdata = sample_boundaries(geo, trn.val_n_boundary_each)
    best_θ = copy(θ)
    best_val = Float32(Inf)
    best_epoch = 0
    stale_epochs = 0
    completed_epochs = 0
    t0 = time()

    if resume && isfile(ckpt_path)
        ckpt = deserialize(ckpt_path)
        if haskey(ckpt, "theta")
            θ .= Float32.(ckpt["theta"])
            if haskey(ckpt, "best_theta")
                best_θ .= Float32.(ckpt["best_theta"])
            else
                best_θ .= θ
            end
            total_hist = Float32.(ckpt["loss_total"])
            pde_hist = Float32.(ckpt["loss_pde"])
            bc_hist = Float32.(ckpt["loss_bc"])
            val_hist = Float32.(ckpt["loss_val"])
            best_epoch = Int(ckpt["best_epoch"])
            best_val = Float32(ckpt["best_val"])
            completed_epochs = haskey(ckpt, "current_epoch") ? Int(ckpt["current_epoch"]) : length(total_hist)
            if haskey(ckpt, "val_interior")
                val_interior = ckpt["val_interior"]
            end
            if haskey(ckpt, "val_bdata")
                val_bdata = ckpt["val_bdata"]
            end
            t0 = time() - max(0, completed_epochs) * 1.0
            println("Resuming from checkpoint: epoch ", completed_epochs, "/", target_epochs,
                    " | best_epoch=", best_epoch, " best_val=", best_val)
        end
    end

    if completed_epochs >= target_epochs
        println("Checkpoint already reached target epochs (", completed_epochs, "). Skipping training.")
        return best_θ, total_hist, pde_hist, bc_hist, val_hist
    end

    adam_remaining = max(0, trn.adam_epochs - completed_epochs)
    if adam_remaining > 0
        start_epoch = completed_epochs + 1
        θ, best_θ, best_val, best_epoch, stale_epochs = train_stage!(
            θ, re, mat, geo, bc, trn;
            epochs=adam_remaining,
            lr=trn.learning_rate,
            start_epoch=start_epoch,
            best_θ=best_θ,
            best_val=best_val,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            total_hist=total_hist,
            pde_hist=pde_hist,
            bc_hist=bc_hist,
            val_hist=val_hist,
            val_interior=val_interior,
            val_bdata=val_bdata,
            t0=t0,
            ckpt_path=ckpt_path,
        )
        completed_epochs = length(total_hist)
    end

    finetune_start = max(completed_epochs + 1, trn.adam_epochs + 1)
    finetune_remaining = max(0, target_epochs - max(completed_epochs, trn.adam_epochs))

    if finetune_remaining > 0 && (trn.early_stop_patience <= 0 || stale_epochs < trn.early_stop_patience)
        println("Starting fine-tune stage with lower LR = ", trn.finetune_lr)
        θ, best_θ, best_val, best_epoch, stale_epochs = train_stage!(
            θ, re, mat, geo, bc, trn;
            epochs=finetune_remaining,
            lr=trn.finetune_lr,
            start_epoch=finetune_start,
            best_θ=best_θ,
            best_val=best_val,
            best_epoch=best_epoch,
            stale_epochs=stale_epochs,
            total_hist=total_hist,
            pde_hist=pde_hist,
            bc_hist=bc_hist,
            val_hist=val_hist,
            val_interior=val_interior,
            val_bdata=val_bdata,
            t0=t0,
            ckpt_path=ckpt_path,
        )
    end

    save_checkpoint(
        ckpt_path,
        θ,
        best_θ,
        total_hist,
        pde_hist,
        bc_hist,
        val_hist,
        best_epoch,
        best_val,
        length(total_hist),
        val_interior,
        val_bdata,
    )
    println("Best validation epoch: ", best_epoch, " | Best validation loss: ", best_val)

    return best_θ, total_hist, pde_hist, bc_hist, val_hist
end

function field_on_grid(model, geo::GeometryParams; nx::Int = 121, ny::Int = 121)
    xs = collect(range(geo.xmin, geo.xmax; length=nx))
    ys = collect(range(geo.ymin, geo.ymax; length=ny))
    Φ = [point_in_notch_void(Float32(x), Float32(y), geo) ? NaN32 : phi_scalar(model, x, y) for y in ys, x in xs]
    return xs, ys, Φ
end

function gradmag_on_line(model, xline::AbstractVector{<:Real}, yline::AbstractVector{<:Real}, geo::GeometryParams; h::Float32 = 2.0f-3)
    return [begin gx, gy = grad_phi_fd(model, Float32(x), Float32(y), geo, h); sqrt(gx^2 + gy^2) end for (x, y) in zip(xline, yline)]
end

function save_plots(model, loss_hist, pde_hist, bc_hist, val_hist, geo::GeometryParams; outdir::String = "results_strainlimiting")
    mkpath(outdir)

    p1 = plot(loss_hist; yscale=:log10, lw=2, label="L total", xlabel="Epoch", ylabel="Loss", title="Training history")
    plot!(p1, pde_hist; lw=2, label="L_pde")
    plot!(p1, bc_hist; lw=2, label="L_bc")
    if !isempty(val_hist)
        plot!(p1, val_hist; lw=2, label="L_val")
    end
    savefig(p1, joinpath(outdir, "loss_history.png"))

    xs, ys, Φ = field_on_grid(model, geo)
    p2 = heatmap(xs, ys, Φ; color=:turbo, xlabel="x", ylabel="y", title="Stress function Φ(x,y)")
    savefig(p2, joinpath(outdir, "phi_field.png"))

    x0, y0 = geo.tip
    xline = collect(range(geo.xmin, x0; length=250))
    yline = fill(y0, length(xline))
    gline = gradmag_on_line(model, xline, yline, geo)
    dist_to_tip = x0 .- xline

    p3 = plot(dist_to_tip, gline; lw=2, xlabel="Distance to notch tip", ylabel="|∇Φ|", title="Gradient magnitude along reference line", label="|∇Φ|")
    savefig(p3, joinpath(outdir, "gradmag_reference_line.png"))
end

function get_run_outdir(root_outdir::String; resume::Bool = false, run_name::AbstractString = "")
    mkpath(root_outdir)
    latest_file = joinpath(root_outdir, "latest_run.txt")

    selected_run = ""
    if !isempty(run_name)
        selected_run = String(run_name)
    elseif resume && isfile(latest_file)
        selected_run = strip(read(latest_file, String))
    else
        selected_run = Dates.format(now(), "yyyymmdd_HHMMSS")
    end

    if isempty(selected_run)
        selected_run = Dates.format(now(), "yyyymmdd_HHMMSS")
    end

    outdir = joinpath(root_outdir, selected_run)
    mkpath(outdir)
    write(latest_file, selected_run * "\n")
    return outdir, selected_run
end

function main()
    mat = MaterialParams()
    geo = GeometryParams()
    bc = BCParams(
        σ0=parse(Float32, get(ENV, "KAN_PINN_SIGMA0", "1.0")),
        L=parse(Float32, get(ENV, "KAN_PINN_L", "1.0")),
    )
    trn = TrainParams(
        adam_epochs=parse(Int, get(ENV, "KAN_PINN_ADAM_EPOCHS", get(ENV, "KAN_PINN_EPOCHS", "1200"))),
        finetune_epochs=parse(Int, get(ENV, "KAN_PINN_FINETUNE_EPOCHS", "300")),
        n_interior_uniform=parse(Int, get(ENV, "KAN_PINN_NU", "96")),
        n_interior_refine=parse(Int, get(ENV, "KAN_PINN_NR", "96")),
        n_interior_tip_strip=parse(Int, get(ENV, "KAN_PINN_NTIP", "128")),
        n_boundary_each=parse(Int, get(ENV, "KAN_PINN_NB", "48")),
        val_n_interior_uniform=parse(Int, get(ENV, "KAN_PINN_VAL_NU", "192")),
        val_n_interior_refine=parse(Int, get(ENV, "KAN_PINN_VAL_NR", "192")),
        val_n_interior_tip_strip=parse(Int, get(ENV, "KAN_PINN_VAL_NTIP", "256")),
        val_n_boundary_each=parse(Int, get(ENV, "KAN_PINN_VAL_NB", "96")),
        learning_rate=parse(Float32, get(ENV, "KAN_PINN_LR", "1.0e-3")),
        finetune_lr=parse(Float32, get(ENV, "KAN_PINN_FINETUNE_LR", "2.0e-4")),
        print_every=parse(Int, get(ENV, "KAN_PINN_PRINT_EVERY", "50")),
        fd_eps=parse(Float32, get(ENV, "KAN_PINN_FD_EPS", "2.0e-3")),
        gc_every=parse(Int, get(ENV, "KAN_PINN_GC_EVERY", "25")),
        early_stop_patience=parse(Int, get(ENV, "KAN_PINN_PATIENCE", "300")),
        min_improve=parse(Float32, get(ENV, "KAN_PINN_MIN_IMPROVE", "1.0e-4")),
        max_grad_norm=parse(Float32, get(ENV, "KAN_PINN_MAX_GRAD_NORM", "10.0")),
        checkpoint_every=parse(Int, get(ENV, "KAN_PINN_CKPT_EVERY", "100")),
        tip_strip_half_height=parse(Float32, get(ENV, "KAN_PINN_TIP_STRIP_HH", "0.035")),
        tip_strip_length=parse(Float32, get(ENV, "KAN_PINN_TIP_STRIP_LEN", "0.20")),
        seed=parse(Int, get(ENV, "KAN_PINN_SEED", "42")),
    )
    resume_training = lowercase(get(ENV, "KAN_PINN_RESUME", "0")) in ("1", "true", "yes", "y")
    run_name = strip(get(ENV, "KAN_PINN_RUN_NAME", ""))

    println("Starting training (Eq. 40 interior + Dirichlet Table-3 BCs on Γ1-Γ4; natural on Γ5a/Γ5b).")

    Random.seed!(trn.seed)

    model0 = build_kan_model(hidden=32, n_basis=16)
    θ0, re = Flux.destructure(model0)
    θ = copy(θ0)

    root_outdir = joinpath(@__DIR__, "results_strainlimiting")
    outdir, selected_run = get_run_outdir(root_outdir; resume=resume_training, run_name=run_name)
    println("Run directory: ", outdir)
    println("Run ID: ", selected_run)
    best_θ, lhist, lpde_hist, lbc_hist, val_hist = train!(θ, re, mat, geo, bc, trn; outdir=outdir, resume=resume_training)
    model = re(best_θ)

    run_cross_verification(model, mat, geo, trn)
    save_plots(model, lhist, lpde_hist, lbc_hist, val_hist, geo; outdir=outdir)
    println("Training complete. Outputs saved in: ", outdir)
end

main()
