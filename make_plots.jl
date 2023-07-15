## USAGE
## -----
##
## From the command line, execute the following command to
## make the plots in the paper after running all experiments:
##     julia make_plots.jl
##
## Edit the 'Paths' variables below as necessary


# Paths
dir_src = "./src/"            # Source code directory
dir_out = "./outputs/"        # Output directory
dir_plt = "./outputs/plots/"  # Output plot directory
dir_dat = "./outputs/data/"   # Cleaned data directory


# Packages and modules
# --------------------

println("Loading packages and modules")

using Random
using Distributions
using HypothesisTests
using DataFrames
using CSV
using Plots
using LaTeXStrings
using Measures
using Colors
using ColorSchemes
using HDF5
import H5Zblosc

include(dir_src * "global_variables.jl")  # Global variables
include(dir_src * "util.jl")              # Shared functions
include(dir_src * "groups.jl")            # Groups and related functions


# Default plot settings and convenience plots
# -------------------------------------------

default(fontfamily="Computer Modern", framestyle=:box, label=nothing, grid=false, legend=:none,
        linewidth=4, titlefontsize=9, guidefontsize=9, tickfontsize=7, legendfontsize=8,
        margin=0mm, dpi=300)

# Set default palette to be CVD-friendly
gr(palette = :tol_bright)
colsDict = Dict(GV_2S=>1, GV_MMD=>2, GV_NMMD=>3, GV_CW=>4, GV_KCI=>5, GV_CP=>6)

# p-value x-label hack
xlab_pv = scatter([0.5], [0], xlims=[0,1], yticks=[], alpha=0, framestyle=:none,
                  series_annotation=text(L"p\mathrm{-value}",:bottom,pointsize=9))

# Blank plot
ep = plot(framestyle=:none, ticks=[])


# Plotting functions
# ------------------

# Plots mean rejection rates and standard deviations
function plot_rej(df::DataFrame, x, regexp="";
                  xlab="", ylab="", ylims=:auto, pl_xticks=false, pl_yticks=false)
    p = plot(xlab=xlab, ylab=ylab, xlims=(minimum(x),maximum(x)), ylims=ylims, grid=true)
    for i in 1:nrow(df)
        rej = collect(df[i,Regex(regexp*"_rej\$")])
        sd = collect(df[i,Regex(regexp*"_rej_sd\$")])
        test = df[i, "Test"]
        plot!(x, rej, linecolor=colsDict[test], linealpha=0.7, label=test)
        plot!(x, max.(0,rej.-sd), fillrange=min.(1,rej.+sd), label=:none, fillalpha=0.2, linealpha=0, fillcolor=colsDict[test])
    end
    if !pl_xticks
        plot!(xticks = (xticks(p)[1][1],""))
    end
    if !pl_yticks
        plot!(yticks = (yticks(p)[1][1],""))
    end
    return p
end

# Plots mean computation time
function plot_time(df::DataFrame, x, regexp="";
                   xlab="", ylab="", ylims=:auto, pl_xticks=false, pl_yticks=false)
    p = plot(xlab=xlab, ylab=ylab, xlims=(minimum(x),maximum(x)), ylims=ylims, grid=true)
    for i in 1:nrow(df)
        time = collect(df[i,Regex(regexp*"_avgtime\$")])
        test = df[i, "Test"]
        plot!(x, time, linecolor=colsDict[test], linealpha=0.7, label=df[i,"Test"])
    end
    if !pl_xticks
        plot!(xticks = (xticks(p)[1][1],""))
    end
    if !pl_yticks
        plot!(yticks = (yticks(p)[1][1],""))
    end
    return p
end

# Plots p-value distribution
function hist(test, x, ylab="";
              bins=10, bw=0.1, p_col=:white, p_y=0, x_ticks=[0,0.5,1], title="", last_row=false, KS=true)
    xt = last_row ? [x_ticks[i]==0 ? "0" : rstrip(string(x_ticks[i]),['0','.']) for i in 1:length(x_ticks)] : []
    xx = vcat(x, collect(minimum(x_ticks):maximum(x_ticks))./bins)  # Visual hack to force all bins to show
    p = histogram(xx, bins=bins, fillcolor=colsDict[test], linecolor=colsDict[test], lw=1, bar_width=bw,
                  xticks=(x_ticks,xt), xlims=[minimum(x_ticks),maximum(x_ticks)], ylab=ylab, yticks=[], title=title)
    yl = ylims(p)
    ylims!(0, yl[2])
    if KS
        # Perform Kolmogorov-Smirnov test and show p-value
        P_U = Uniform(0, 1)
        n = length(x)
        pv = pvalue(ExactOneSampleKSTest(jitter(x), P_U))
        scatter!([0.77], [p_y], alpha=0,
                 series_annotation=text(L"$\mathbf{%$(rpad(round(pv,digits=3),5,'0'))}$",:bottom,pointsize=9,color=p_col))
    end
    return p
end


# Invariant distribution example plots
# ------------------------------------

# Set seed for reproducibility
Random.seed!(1)

# Make contour plots
N = 100
tick0 = [0]

# Standard Gaussian
P_sg = MvNormal([0,0], 1)
x_sg = range(-2.75, 2.75, length=N)
y_sg = range(-2.75, 2.75, length=N)
f_sg = (x,y) -> pdf(P_sg, vcat(x,y))
z_sg = @. f_sg(x_sg',y_sg)
pl_sg = contourf(x_sg, y_sg, z_sg, levels=20, lw=0, xticks=(tick0,[]), yticks=tick0, color=:Blues, cbar=false, ylab="Density")
vline!([0], color=:black, alpha=0.1, lw=1)
hline!([0], color=:black, alpha=0.1, lw=1)

# Chi * von Mises
P_c = Chi(2)
P_vM = VonMises(π/4, 4)
x_cvM = range(-2.75, 2.75, length=N)
y_cvM = range(-2.75, 2.75, length=N)
f_cvM = (x,y) -> begin
    r = sqrt(x^2 + y^2)
    return pdf(P_c,r) * pdf(P_vM,atan(y,x)) / r
end
z_cvM = @. f_cvM(x_cvM',y_cvM)
pl_cvM = contourf(x_cvM, y_cvM, z_cvM, levels=20, lw=0, xticks=(tick0,[]), yticks=(tick0,[]), color=:Oranges, cbar=false)
vline!([0], color=:black, alpha=0.1, lw=1)
hline!([0], color=:black, alpha=0.1, lw=1)

# Orbit-averaged chi
P_u = Uniform(0, 2*π)
x_g = range(-2.75, 2.75, length=N)
y_g = range(-2.75, 2.75, length=N)
f_g = (x,y) -> begin
    r = sqrt(x^2 + y^2)
    return pdf(P_c,r) * pdf(P_u,abs(atan(y,x))) / r
end
z_g = @. f_g(x_g',y_g)
pl_g = contourf(x_g, y_g, z_g, levels=20, lw=0, xticks=(tick0,[]), yticks=(tick0,[]), color=:Greens, cbar=false)
vline!([0], color=:black, alpha=0.1, lw=1)
hline!([0], color=:black, alpha=0.1, lw=1)

# Make sample plots
N = 50
x_lims = (-4, 4)
y_lims = (-4, 4)

x_sg = rand(P_sg, N)
sc_sg = scatter(x_sg[1,:], x_sg[2,:], xlims=x_lims, ylims=y_lims, xticks=tick0, yticks=tick0,
                markercolor=palette(:Blues)[6], markerstrokewidth=0, alpha=0.7, ylab="Samples")
vline!([0], color=:black, alpha=0.1, lw=1)
hline!([0], color=:black, alpha=0.1, lw=1)

x_c = vcat(rand(P_c,N)', fill(0,N)')

x_cvM = Matrix{Float64}(undef, 2, N)
θ_vM = rand(P_vM, N)
for i in 1:N
    x_cvM[:,i] = rotate_2D(x_c[:,i], θ_vM[i])
end
sc_cvM = scatter(x_cvM[1,:], x_cvM[2,:], xlims=x_lims, ylims=y_lims, xticks=tick0, yticks=(tick0,[]),
    markercolor=palette(:Oranges)[6], markerstrokewidth=0, alpha=0.7)
vline!([0], color=:black, alpha=0.1, lw=1)
hline!([0], color=:black, alpha=0.1, lw=1)

x_g = Matrix{Float64}(undef, 2, N)
θ_u = rand(P_u, N)
for i in 1:N
    x_g[:,i] = rotate_2D(x_c[:,i], θ_u[i])
end
sc_g = scatter(x_g[1,:], x_g[2,:], xlims=x_lims, ylims=y_lims, xticks=tick0, yticks=(tick0,[]),
    markercolor=palette(:Greens)[6], markerstrokewidth=0, alpha=0.7)
vline!([0], color=:black, alpha=0.05, lw=1)
hline!([0], color=:black, alpha=0.05, lw=1)

# Make representative inversion plots
y_lims = [0, 2*π]
x_lims = [1, 50]

θ_sg = zeros(N)
for i in 1:N
    θ_sg[i] = tau_inv_rotate_2D(x_sg[:,i])
end
cdf_sg = plot(1:N, sort(θ_sg), linecolor=palette(:Blues)[6], grid=true,
              xlims=x_lims, ylims=y_lims, ylab=L"Sorted $\tau(X)^{-1}$", xticks=tick0)

θ_cvM = zeros(N)
for i in 1:N
    θ_cvM[i] = tau_inv_rotate_2D(x_cvM[:,i])
end
cdf_cvM = plot(1:N, sort(θ_cvM), linecolor=palette(:Oranges)[6], xlims=x_lims, ylims=y_lims, xticks=tick0, grid=true)

θ_g = zeros(N)
for i in 1:N
    θ_g[i] = tau_inv_rotate_2D(x_g[:,i])
end
cdf_g = plot(1:N, sort(θ_g), linecolor=palette(:Greens)[6], xlims=x_lims, ylims=y_lims, xticks=tick0, grid=true)

fig = plot(pl_sg, pl_cvM, pl_g,
           sc_sg, sc_cvM, sc_g,
           cdf_sg, cdf_cvM, cdf_g,
           layout=grid(3,3,heights=[0.38,0.38,0.24]), size=(525,425))
fig_name_pdf = dir_plt * "densities.pdf"
fig_name_png = dir_plt * "densities.png"  # Note: manually convert this to PDF due to issue with savefig and PDFs
savefig(fig, fig_name_pdf)
println("Created $(fig_name_pdf)")
savefig(fig, fig_name_png)
println("Created $(fig_name_png)")


# SO(4) experiment: increasing parameters
# ---------------------------------------

# Read experiment results
df_d = CSV.read(dir_out*"invariance_rotation_dvar_N1000_n200_M2_B200.csv", DataFrame)
df_n = CSV.read(dir_out*"invariance_rotation_d4_N1000_nvar_M2_200.csv", DataFrame)
df_M = CSV.read(dir_out*"invariance_rotation_d4_N1000_n200_Mvar_B200.csv", DataFrame)

# Make plots
H0_lims = [0, 0.10]
H1_lims = [0, 1]
time_lims = [0, 8]

ds = [5,10,15,20]
fig_d0 = plot_rej(df_d, ds, "H0.*"; ylab=L"$H_0$ rej. rate", ylims=H0_lims, pl_yticks=true)
fig_d1 = plot_rej(df_d, ds, "H1.*", ylab=L"$H_1$ rej. rate", ylims=H1_lims, pl_yticks=true)
fig_dt = plot_time(df_d, ds, "H1.*", xlab="Dimensions", ylab="Avg. time (s)", ylims=time_lims, pl_xticks=true, pl_yticks=true)

ns = [50,100,200,400]
fig_n0 = plot_rej(df_n, ns, "H0.*", ylims=H0_lims)
fig_n1 = plot_rej(df_n, ns, "H1.*", ylims=H1_lims)
fig_nt = plot_time(df_n, ns, "H1.*", xlab="Sample size", ylims=time_lims, pl_xticks=true)

Ms = collect(1:5)
fig_M0 = plot_rej(df_M, Ms, "H0.*", ylims=H0_lims)
fig_M1 = plot_rej(df_M, Ms, "H1.*", ylims=H1_lims)
fig_Mt = plot_time(df_M, Ms, "H1.*", xlab="Group actions", ylims=time_lims, pl_xticks=true)

legend = plot([0 0 0 0], showaxis=false, legend=true, label=reshape(df_d[:,"Test"],1,4), legendcolumns=4,
              foreground_color_legend=nothing, color=[colsDict[GV_2S] colsDict[GV_MMD] colsDict[GV_NMMD] colsDict[GV_CW]])
l = @layout [a{0.01h} ; grid(3,3)]
fig = plot(legend,
           fig_d0, fig_n0, fig_M0,
           fig_d1, fig_n1, fig_M1,
           fig_dt, fig_nt, fig_Mt,
           layout=l, size=(600,375))
fig_name = dir_plt * "invariance_so4.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# Exchangeability experiment: increasing number of random projections
# -------------------------------------------------------------------

# Read experiment results
df = CSV.read(dir_out*"invariance_exchangeability_d10_N1000_n200_M2_B200_Jvar.csv", DataFrame)

# Make plot
Js = collect(5:5:50)

fig_J0p = plot_rej(df, Js, "H0\\+.*", ylims=[0,0.1], ylab="Rej. rate", pl_yticks=true)
fig_J0m = plot_rej(df, Js, "H0-.*", ylims=[0,0.1], pl_yticks=true)
fig_J1 = plot_rej(df, Js, "H1.*", ylims=[0,1], pl_yticks=true)

fig_Jt_0p = plot_time(df, Js, "H0\\+.*", ylab="Avg. time (s)", pl_xticks=true, pl_yticks=true)
fig_Jt_0m = plot_time(df, Js, "H0-.*", xlab="Rand. projections", pl_xticks=true, pl_yticks=true)
fig_Jt_1 = plot_time(df, Js, "H1.*", pl_xticks=true, pl_yticks=true)

legend = plot([0 0], legend=true, showaxis=false, label=reshape(df[:,"Test"],1,2), legendcolumns=2,
              foreground_color_legend=nothing, color=[colsDict[GV_NMMD] colsDict[GV_CW]])

l = @layout[grid(2,3) ; a{0.01h}]
fig = plot(fig_J0p, fig_J0m, fig_J1,
           fig_Jt_0p, fig_Jt_0m, fig_Jt_1,
           legend, title=[L"H_0^+" L"H_0^-" L"H_1" "" "" "" ""], top_margin=1mm, layout=l, size=(525,300))
fig_name = dir_plt * "invariance_exch.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# SO(4) and exchangeability invariance p-values
# ---------------------------------------------

# Read experiment results
pv_rot = CSV.read(dir_out*"invariance_rotation_d4_N1000_n200_M2_B200.csv", DataFrame)
pv_exch = CSV.read(dir_out*"invariance_exchangeability_d10_N1000_n200_M2_B200.csv", DataFrame)

# Set seed for reproducibility
Random.seed!(1)

# Make plot
r01 = hist(GV_2S, pv_rot[!,:H0_2SMMD_p], "2SMMD", title=L"Rot. $H_0$")
r02 = hist(GV_MMD, pv_rot[!,:H0_MMD_p], "MMD")
r03 = hist(GV_NMMD, pv_rot[!,:H0_NMMD_p], "NMMD")
r04 = hist(GV_CW, pv_rot[!,:H0_CW_p], "CW", last_row=true, p_col=:grey15)

r11 = hist(GV_2S, pv_rot[!,:H1_2SMMD_p], title=L"Rot. $H_1$", p_col=:grey15)
r12 = hist(GV_MMD, pv_rot[!,:H1_MMD_p], p_col=:grey15)
r13 = hist(GV_NMMD, pv_rot[!,:H1_NMMD_p], p_col=:grey15)
r14 = hist(GV_CW, pv_rot[!,:H1_CW_p], last_row=true, p_col=:grey15)

e01p = hist(GV_2S, pv_exch[!,"H0+_2SMMD_p"], title=L"Exch. $H_0^+$")
e02p = hist(GV_MMD, pv_exch[!,"H0+_MMD_p"])
e03p = hist(GV_NMMD, pv_exch[!,"H0+_NMMD_p"])
e04p = hist(GV_CW, pv_exch[!,"H0+_CW_p"], last_row=true, p_col=:grey15)

e01m = hist(GV_2S, pv_exch[!,"H0-_2SMMD_p"], title=L"Exch. $H_0^-$")
e02m = hist(GV_MMD, pv_exch[!,"H0-_MMD_p"])
e03m = hist(GV_NMMD, pv_exch[!,"H0-_NMMD_p"])
e04m = hist(GV_CW, pv_exch[!,"H0-_CW_p"], last_row=true, p_col=:grey15)

e11 = hist(GV_2S, pv_exch[!,"H1_2SMMD_p"], title=L"Exch. $H_1$", p_col=:grey15)
e12 = hist(GV_MMD, pv_exch[!,"H1_MMD_p"], p_col=:grey15)
e13 = hist(GV_NMMD, pv_exch[!,"H1_NMMD_p"], p_y=50, p_col=:grey15)
e14 = hist(GV_CW, pv_exch[!,"H1_CW_p"], last_row=true, p_col=:grey15)

l = @layout [grid(4,5) ; a{0.02h}]
fig = plot(r01, r11, e01p, e01m, e11,
           r02, r12, e02p, e02m, e12,
           r03, r13, e03p, e03m, e13,
           r04, r14, e04p, e04m, e14,
           xlab_pv, layout=l, size=(550,400), left_margin=2mm)
fig_name = dir_plt * "invariance_so4_exch_pvals.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# SO(3) invariance p-values
# -------------------------

# Read experiment results
pv_so3 = CSV.read(dir_out*"invariance_SO3_N1000_n100_M2_B200.csv", DataFrame)

# Set seed for reproducibility
Random.seed!(1)

# Make plot
s01 = hist(GV_2S, pv_so3[!,:H0_2SMMD_p], "2SMMD", title=L"H_0")
s02 = hist(GV_MMD, pv_so3[!,:H0_MMD_p], "MMD")
s03 = hist(GV_NMMD, pv_so3[!,:H0_NMMD_p], "NMMD", last_row=true)

s11 = hist(GV_2S, pv_so3[!,:H1_2SMMD_p], title=L"H_1", p_col=:grey15)
s12 = hist(GV_MMD, pv_so3[!,:H1_MMD_p], p_col=:grey15)
s13 = hist(GV_NMMD, pv_so3[!,:H1_NMMD_p], last_row=true, p_col=:grey15, p_y=55)

l = @layout [grid(3,2) ; a{0.02h}]
fig = plot(s01, s11,
           s02, s12,
           s03, s13,
           xlab_pv, layout=l, size=(240,330))
fig_name = dir_plt * "invariance_so3_pvals.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# SWARM data visualization
# ------------------------

# Read in data
fid = h5open(dir_out_dat*"SWARM.h5", "r")
SWARM = copy(read(fid["data"]))
close(fid)
n_SWARM = size(SWARM, 2)
n = 220

function SWARM_sample_data(n)
    data = SWARM
    n_data = n_SWARM
    inds = sample(1:n_data, n, replace=false)
    x = data[1:3, inds]
    y = reshape(data[4,inds], 1, n)
    return OneSampleData(x=x, y=y)
end

# Set seed for reproducibility
Random.seed!(1)
dat = SWARM_sample_data(n)
x = dat.x
y = dat.y

# Make plot
cols = cgrad([:orchid1, :midnightblue])
p_xy = scatter(x[1,:],x[2,:], alpha=0.7, c=cols)
p_xz = scatter(x[1,:],x[3,:], alpha=0.7, c=cols)
p_yz = scatter(x[2,:],x[3,:], alpha=0.7, c=cols)

fig = plot(p_xy, p_xz, p_yz,
           layout=(1,3), title=[L"X_1:X_2" L"X_1:X_3" L"X_2:X_3"], framestyle=:none, size=(500,150), margin=2mm,
           zcolor=y[:], markerstrokewidth=0, cbar=false, xticks=[], yticks=[])
fig_name = dir_plt * "SWARM_data.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# SWARM conditional invariance p-value distributions
# --------------------------------------------------

# Read experiment results
pv_swarm = CSV.read(dir_out*"SWARM_conditional_invariance_N1000_n220_B200.csv", DataFrame)

# Make plot
pX = hist(GV_KCI, pv_swarm[!,:X_inv_KCI_p], "KCI", title=L"$X_1$-axis", x_ticks=[0,0.25,0.5], last_row=true, bw=0.05, KS=false)
pY = hist(GV_KCI, pv_swarm[!,:Y_inv_KCI_p], title=L"$X_2$-axis", x_ticks=[0,0.25,0.5], last_row=true, bw=0.05, bins=20, KS=false)
pZ = hist(GV_KCI, pv_swarm[!,:Z_inv_KCI_p], title=L"$X_3$-axis", x_ticks=[0,0.25,0.5], last_row=true, bw=0.05, KS=false)
vline!([0.075], color=:darkorange, lw=1.5)
pNP = hist(GV_KCI, pv_swarm[!,:NP_KCI_p], title="geomag. NP", x_ticks=[0,0.25,0.5], last_row=true, bw=0.05, KS=false)

l = @layout [grid(1,4) ; a{0.1h}]
fig = plot(pX, pY, pZ, pNP,
           xlab_pv, layout=l, size=(500,140), top_margin=1mm, left_margin=3.5mm)
fig_name = dir_plt * "SWARM_cond_indep_pvals.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# SWARM invariance p-value distributions
# --------------------------------------

# Read experiment results
pv_swarm_inv = CSV.read(dir_out*"SWARM_invariance_N1000_n220_B200.csv", DataFrame)
pv_swarm_jinv = CSV.read(dir_out*"SWARM_joint_invariance_N1000_n220_B200.csv", DataFrame)

# Set seed for reproducibility
Random.seed!(1)

# Make plot
pd_2S = hist(GV_2S, pv_swarm_inv[!,:inv_disc_2SMMD_p], "2SMMD", title="Marg. + Disc.")
pd_MMD = hist(GV_MMD, pv_swarm_inv[!,:inv_disc_MMD_p], "MMD")
pd_NMMD = hist(GV_NMMD, pv_swarm_inv[!,:inv_disc_NMMD_p], "NMMD")
pd_CW = hist(GV_CW, pv_swarm_inv[!,:inv_disc_CW_p], "CW", last_row=true, p_col=:grey15)

pc_2S = hist(GV_2S, pv_swarm_inv[!,:inv_cont_2SMMD_p], title="Marg. + Cont.")
pc_MMD = hist(GV_MMD, pv_swarm_inv[!,:inv_cont_MMD_p])
pc_NMMD = hist(GV_NMMD, pv_swarm_inv[!,:inv_cont_NMMD_p])
pc_CW = hist(GV_CW, pv_swarm_inv[!,:inv_cont_CW_p], last_row=true, p_col=:grey15)

pjd_2S = hist(GV_2S, pv_swarm_jinv[!,:jinv_disc_2SMMD_p], title="Joint + Disc.", p_col=:grey15, p_y=150)
pjd_MMD = hist(GV_MMD, pv_swarm_jinv[!,:jinv_disc_MMD_p], p_col=:grey15)
pjd_NMMD = hist(GV_NMMD, pv_swarm_jinv[!,:jinv_disc_NMMD_p], p_col=:grey15, p_y=138)
pjd_CW = hist(GV_CW, pv_swarm_jinv[!,:jinv_disc_CW_p], last_row=true, p_col=:grey15)

pjc_2S = hist(GV_2S, pv_swarm_jinv[!,:jinv_cont_2SMMD_p], title="Joint + Cont.", p_col=:grey15, p_y=145)
pjc_MMD = hist(GV_MMD, pv_swarm_jinv[!,:jinv_cont_MMD_p], p_col=:grey15)
pjc_NMMD = hist(GV_NMMD, pv_swarm_jinv[!,:jinv_cont_NMMD_p], p_col=:grey15, p_y=147)
pjc_CW = hist(GV_CW, pv_swarm_jinv[!,:jinv_cont_CW_p], last_row=true, p_col=:grey15)

l = @layout [grid(4,4) ; a{0.02h}]
fig = plot(pd_2S, pc_2S, pjd_2S, pjc_2S,
           pd_MMD, pc_MMD, pjd_MMD, pjc_MMD,
           pd_NMMD, pc_NMMD, pjd_NMMD, pjc_NMMD,
           pd_CW, pc_CW, pjd_CW, pjc_CW,
           xlab_pv, layout=l, size=(450,420), left_margin=1mm)
fig_name = dir_plt * "SWARM_invariance_pvals.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# LHC p-value distributions
# -------------------------

# Read experiment results
pv_lhc_inv = CSV.read(dir_out*"LHC_invariance_N1000_n100_M2_B200.csv", DataFrame)
pv_lhc_eq = CSV.read(dir_out*"LHC_equivariance_N1000_n100_B200.csv", DataFrame)

# Set seed for reproducibility
Random.seed!(1)

# Make plots
pH0_2S = hist(GV_2S, pv_lhc_inv[!,:H0_2SMMD_p], "2SMMD", title=L"\mathbf{G}_0")
pH0_MMD = hist(GV_MMD, pv_lhc_inv[!,:H0_MMD_p], "MMD")
pH0_NMMD = hist(GV_NMMD, pv_lhc_inv[!,:H0_NMMD_p], "NMMD")
pH0_CW = hist(GV_CW, pv_lhc_inv[!,:H0_CW_p], "CW", last_row=true)

pH1_ind_2S = hist(GV_2S, pv_lhc_inv[!,:H1_ind_2SMMD_p], title=L"\mathbf{G}_1", p_col=:grey15)
pH1_ind_MMD = hist(GV_MMD, pv_lhc_inv[!,:H1_ind_MMD_p], p_col=:grey15)
pH1_ind_NMMD = hist(GV_NMMD, pv_lhc_inv[!,:H1_ind_NMMD_p], p_col=:grey15)
pH1_ind_CW = hist(GV_CW, pv_lhc_inv[!,:H1_ind_CW_p], last_row=true, p_col=:grey15)

pH1_so4_2S = hist(GV_2S, pv_lhc_inv[!,:H1_SO4_2SMMD_p], title=L"\mathbf{G}_2", p_col=:grey15)
pH1_so4_MMD = hist(GV_MMD, pv_lhc_inv[!,:H1_SO4_MMD_p], p_col=:grey15)
pH1_so4_NMMD = hist(GV_NMMD, pv_lhc_inv[!,:H1_SO4_NMMD_p], p_col=:grey15)
pH1_so4_CW = hist(GV_CW, pv_lhc_inv[!,:H1_SO4_CW_p], last_row=true, p_col=:grey15)

pH0_KCI = hist(GV_KCI, pv_lhc_eq[!,:H0_KCI_p], "KCI", title=L"SO$(2)$-equiv.", p_col=:grey15)
pH0_CP = hist(GV_CP, pv_lhc_eq[!,:H0_CP_p], "CP", last_row=true)

pH1_KCI = hist(GV_KCI, pv_lhc_eq[!,:H1_KCI_p], title=L"Cond. SO$(2)$-inv.", p_col=:grey15)
pH1_CP = hist(GV_CP, pv_lhc_eq[!,:H1_CP_p], last_row=true, p_col=:grey15)

l = @layout [[grid(4,3) ; b{0.05h}] c{0.001w} [a{0.64w,0.46h} ; grid(2,2) ; d{0.05h}]]
fig = plot(pH0_2S, pH1_ind_2S, pH1_so4_2S,
           pH0_MMD, pH1_ind_MMD, pH1_so4_MMD,
           pH0_NMMD, pH1_ind_NMMD, pH1_so4_NMMD,
           pH0_CW, pH1_ind_CW, pH1_so4_CW,
           xlab_pv, ep, ep,
           pH0_KCI, pH1_KCI,
           pH0_CP, pH1_CP,
           xlab_pv, layout=l, size=(750,415), left_margin=4mm)
fig_name = dir_plt * "LHC_pvals.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# LHC invariance experiment: increasing number of random projections
# ------------------------------------------------------------------

# Read experiment results
df = CSV.read(dir_out*"LHC_invariance_N1000_n100_M2_B200_Jvar.csv", DataFrame)

# Make plot
Js = collect(2:2:20)

fig_J0 = plot_rej(df, Js, "H0.*", ylims=[0,0.1], ylab="Rej. rate", pl_yticks=true)
fig_J1_ind = plot_rej(df, Js, "H1_ind.*", ylims=[0,1], pl_yticks=true)
fig_J1_SO4 = plot_rej(df, Js, "H1_SO4.*", ylims=[0,1], pl_yticks=true)

fig_Jt_0 = plot_time(df, Js, "H0.*", xlab="Rand. projections", ylab="Avg. time (s)", pl_xticks=true, pl_yticks=true)
fig_Jt_ind = plot_time(df, Js, "H1_ind.*", xlab="Rand. projections", pl_xticks=true, pl_yticks=true)
fig_Jt_SO4 = plot_time(df, Js, "H1_SO4.*", xlab="Rand. projections", pl_xticks=true, pl_yticks=true)

legend = plot([0 0], legend=true, showaxis=false, label=reshape(df[:,"Test"],1,2), legendcolumns=2,
              foreground_color_legend=nothing, color=[colsDict[GV_NMMD] colsDict[GV_CW]])
l = @layout[grid(2,3) ; a{0.01h}]
fig = plot(fig_J0, fig_J1_ind, fig_J1_SO4,
           fig_Jt_0, fig_Jt_ind, fig_Jt_SO4,
           legend, layout=l, title=[L"\mathbf{G}_0" L"\mathbf{G}_1" L"\mathbf{G}_2" "" "" "" ""], top_margin=1mm, size=(525,300))
fig_name = dir_plt * "LHC.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")


# Top quark conditional invariance p-value distributions
# ------------------------------------------------------

# Read experiment results
pv_tqt = CSV.read(dir_out*"TQT_conditional_invariance_N1000_n100_B200.csv", DataFrame)

# Make plot
pH0_KCI = hist(GV_KCI, pv_tqt[!,:H0_KCI_p], "KCI", title=L"H_0", KS=false)
pH1_KCI = hist(GV_KCI, pv_tqt[!,:H1_KCI_p], title=L"H_1", KS=false)

l = @layout [grid(1,2) ; a{0.05h}]
fig = plot(pH0_KCI, pH1_KCI,
           xlab_pv, layout=l, size=(240,140))
fig_name = dir_plt * "TQT_pvals.pdf"
savefig(fig, fig_name)
println("Created $(fig_name)")