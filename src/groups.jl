## Collection of group implementations that are tested in the experiments
##
## The following group functions may need to be implemented depending on the test:
##     - f_sample
##         Inputs: none
##         Output: action g
##     - f_transform
##         Inputs: data point x, action g
##         Output: transformed data point gx
##     - f_max_inv
##         Inputs: data matrix X
##         Output: maximal invariant matrix M(X)
##     - f_tau_inv
##         Inputs: data point x
##         Output: action g such that g*x=γ(x)
##     - f_get_generators (only used for distribution-free CW test)
##         Inputs: group-specific
##         Output: generators of group [g1, ..., gm]


using Random
using LinearAlgebra
using Rotations
using RandomMatrices
using InvertedIndices
using SliceMap
using Base.Threads
include("./util.jl")


mutable struct GroupSampler
    M::Int64                    # Number of group actions to sample in tests if needed
    f_sample::Function          # Samples a random group action
    f_transform::Function       # Transforms a data point by an action
    f_get_generators::Function  # Returns generators of group
    f_max_inv::Function         # Applies a maximal invariant to all data points
    f_tau_inv::Function         # Computes the inverse representative inversion
    dz::Int64                   # Dimension of maximal invariant if different from data
    function GroupSampler(M=0; dz=0, f_sample=f_nothing, f_transform=f_nothing,
                          f_get_generators=f_nothing, f_max_inv=f_nothing, f_tau_inv=f_nothing)
        return new(M, f_sample, f_transform, f_get_generators, f_max_inv, f_tau_inv, dz)
    end
end


# Applies a single transformation to an entire dataset
# or a random transformation to each point if no transformation specified
function transform_all(GS::GroupSampler, x::AbstractMatrix{Float64}, g::Any=nothing)
    if isnothing(g)
        return tmapcols(x->GS.f_transform(x,GS.f_sample()), x)
    end
    return tmapcols(x->GS.f_transform(x,g), x)
end


# Transforms each data point in a dataset given a set of predetermined transformations
function transform_each(GS::GroupSampler, x::AbstractMatrix{Float64}, g::AbstractArray{Any,1})
    d, n = size(x)
    gx = Matrix{Float64}(undef, d, n)
    @threads for i in 1:n
        gx[:,i] = @views GS.f_transform(x[:,i],g[i])
    end
    return gx
end


# Identity group
# --------------

# f_sample
function rand_identity()
    return 1
end
# f_transform
function identity(x, g)
    return x
end
# f_max_inv
function max_inv_identity(x)
    return x
end
# f_tau_inv
function tau_inv_identity(x)
    return 1
end
ID_GS = GroupSampler(1, f_sample=rand_identity, f_transform=identity, f_max_inv=max_inv_identity, f_tau_inv=tau_inv_identity)


# Rotations
# ---------

## SO(2)
# f_sample
# Returns rotation angle in [0,2π]
function rand_θ()
    return rand() * 2π
end
# f_transform
function rotate_2D(x::AbstractVector{Float64}, θ::Float64)
    return [x[1]*cos(θ) - x[2]*sin(θ); x[1]*sin(θ) + x[2]*cos(θ)]
end
# f_tau_inv
# Computes rotation angle in [0,2π] that takes x -> [norm(x) 0]
function tau_inv_rotate_2D(x::AbstractVector{Float64}; dim::Int64=1)
    y = orb_rep_rotate(x, 2, dim=dim)
    # https://www.mathworks.com/matlabcentral/answers/180131-how-can-i-find-the-angle-between-two-vectors-including-directional-information
    θ = atan(x[1]*y[2]-x[2]*y[1], x[1]*y[1]+x[2]*y[2])
    # Return counterclockwise angle
    return θ >= 0 ? θ : 2*π+θ
end
# f_max_inv
# function max_inv_rotate(x::AbstractMatrix{Float64}; dim::Int64=1)

## SO(3)
# f_sample
# Returns 3D rotation matrix (optionally as a quarternion)
function rand_3D(;quat=false)
    R = rand(QuatRotation)
    return quat ? mat_to_quat(R) : R
end
# f_transform
function rotate_3D(x, q; quat=false)
    if quat
        return quat_prod(q, x)
    end
    return q * x
end
# f_max_inv
# function max_inv_rotate(x::AbstractMatrix{Float64}; dim::Int64=1)
# f_tau_inv
# function tau_inv_rotate(x::AbstractVector{Float64}; dim::Int64=1)

## 2D rotation about an axis in 3D
# f_sample
# Returns 3D rotation matrix
function rand_3D_axis(;ax::Int64=1)
    R = zeros(3, 3)
    R[ax,ax] = 1
    θ = rand_θ()
    R[Not(ax),Not(ax)] = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    return R
end
# Computes an orbit representative
function orb_rep_rotate_axis(x::AbstractVector{Float64}; ax::Int64=1)
    dim = (ax % 3) + 1
    z = zeros(3)
    z[ax] = x[ax]
    z[dim] = @views norm(x[Not(ax)])
    return z
end
# f_max_inv
function max_inv_rotate_axis(x::AbstractMatrix{Float64}; ax::Int64=1)
    dim = (ax % 3) + 1
    return tmapcols(x->orb_rep_rotate_axis(x,ax=ax), x)
end
# f_tau_inv
function tau_inv_rotate_axis(x::AbstractVector{Float64}; ax::Int64=1)
    dim = (ax % 3) + 1
    
    # Target vector
    ys = orb_rep_rotate_axis(x, ax=ax)[Not(ax)]
    
    R = zeros(3, 3)
    R[ax,ax] = 1
    
    xs = x[Not(ax)]
    norm_xs = norm(xs)
    
    # https://math.stackexchange.com/questions/598750/finding-the-rotation-matrix-in-n-dimensions
    u = xs / norm_xs
    v = ys - dot(u,ys)*u
    v = v / norm(v)
    
    uv = hcat(u, v)
    θ = acos(dot(xs,ys) / norm_xs^2)
    Rs = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    R[Not(ax),Not(ax)] = I(2) - u*u' - v*v' + uv*Rs*uv'
    return R
end

## 2D rotation about north geomagnetic pole
# f_sample: not implemented
# f_transform: not implemented
# f_max_inv
function max_inv_geoNP(x::AbstractMatrix{Float64})
    n = size(x, 2)
    V = [0 0 GV_NPx; 0 0 GV_NPy; -GV_NPx -GV_NPy 0]
    R = I(3) - V + V*V./(1+GV_NPz)
    return R' * max_inv_rotate_axis(R*x,ax=3)
end
# f_tau_inv: not implemented

## 2D rotation about north geographic pole by 0,...,14 * 24 degrees
# f_sample
# Returns rotation angle in [0,2π]
function rand_SWARM_disc_rot()
    return sample(0:14) * 24 / 180 * π
end
# f_transform
# Usable for any data with dim > 2
function rotate_SWARM_rot(x::AbstractVector{Float64}, θ_x::Float64)
    Rx = copy(x)
    Rx[1] = x[1]*cos(θ_x) - x[2]*sin(θ_x)
    Rx[2] = x[1]*sin(θ_x) + x[2]*cos(θ_x)
    return Rx
end
# f_max_inv: not implemented
# f_tau_inv: not implemented

## d-dimensional rotation
# f_sample
# Returns d-dimensional rotation matrix
function rand_rotation(d::Int64)
    R = rand(Haar(1), d)
    if det(R) < 0
        R[[1,2],:] = R[[2,1],:]
    end
    return R
end
# f_transform
function rotate_d(x::AbstractVector{Float64}, R::Matrix{Float64})
    return R * x
end
# Computes an orbit representative
function orb_rep_rotate(x::AbstractVector{Float64}, d::Int64; dim::Int64=1)
    z = zeros(d)
    z[1] = norm(x)
    return z
end
# f_max_inv
function max_inv_rotate(x::AbstractMatrix{Float64}; dim::Int64=1)
    d = size(x, 1)
    return tmapcols(x->orb_rep_rotate(x,d,dim=dim), x)
end
# f_tau_inv
# Computes a rotation matrix that takes x -> [norm(x) 0 ... 0]
function tau_inv_rotate(x::AbstractVector{Float64}; dim::Int64=1)
    d = length(x)
    norm_x = norm(x)
    
    # Target vector
    y = orb_rep_rotate(x, d, dim=dim)
    
    # https://math.stackexchange.com/questions/598750/finding-the-rotation-matrix-in-n-dimensions
    u = x / norm_x
    v = -u[dim] * x
    v[dim] += norm_x
    v = v / norm(v)
    
    uv = hcat(u, v)
    θ = acos(dot(x,y) / norm_x^2)
    Rxy = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    R1 = I(d) - u*u' - v*v' + uv*Rxy*uv'
    
    # Sample and apply a rotation from the stabilizer subgroup
    R2 = diagm(ones(d))
    R2[Not(dim),Not(dim)] = rand_rotation(d-1)
    return R2 * R1
end

## SO(2) x SO(2)
# f_sample
# Returns a pair of angles; if paired=true, the angles are the same
function rand_θ1_θ2(;paired::Bool=true)
    if !paired
        return [rand_θ(), rand_θ()]
    end
    θ = rand_θ()
    return [θ, θ]
end
# f_transform
function rotate_θ1_θ2(x::AbstractVector{Float64}, θs::Vector{Float64})
    R = zeros(4, 4)
    θ1 = θs[1]
    θ2 = θs[2]
    R[1:2,1:2] = [cos(θ1) -sin(θ1); sin(θ1) cos(θ1)]
    R[3:4,3:4] = [cos(θ2) -sin(θ2); sin(θ2) cos(θ2)]
    return R * x
end
# f_max_inv
function max_inv_θ1_θ2(x::AbstractMatrix{Float64}; dim::Int64=1, paired::Bool=true)
    n = size(x, 2)
    max_x = Matrix{Float64}(undef, 4, n)
    max_x[1:2,:] = @views max_inv_rotate(x[1:2,:], dim=dim)
    if paired
        @threads for i in 1:n
            R = @views tau_inv_rotate(x[1:2,i], dim=dim)
            max_x[3:4,i] = R * x[3:4,i]
        end
    else
        max_x[3:4,:] = @views max_inv_rotate(x[3:4,:], dim=dim)
    end
    return max_x
end
# f_tau_inv
function tau_inv_θ1_θ2(x::AbstractVector{Float64}; dim::Int64=1, paired::Bool=true)
    R = zeros(4, 4)
    R[1:2,1:2] = @views tau_inv_rotate(x[1:2], dim=dim)
    if paired
        R[3:4,3:4] = R[1:2,1:2]
    else
        R[3:4,3:4] = @views tau_inv_rotate(x[3:4], dim=dim)
    end
    return R
end


# Exchangeability
# ---------------

# f_sample: randperm(Int)
# f_transform
function permute(x::AbstractVector{Float64}, P::Vector{Int64})
    return x[P]
end
# f_get_generators
function get_permute_generators(d::Int64)
    g1 = collect(1:d)
    g1[1:2] = [2, 1]
    g2 = pushfirst!(collect(1:(d-1)), d)
    return [g1, g2]
end
# f_max_inv
function max_inv_permute(x::AbstractMatrix{Float64})
    d = size(x, 1)
    return tmapcols(x->sort(x), x)
end
# f_tau_inv
# Computes the permutation that takes x -> [x_(1) ... x_(d)]
function tau_inv_permute(x::AbstractVector{Float64})
    return sortperm(x)
end


# Lorentz
# -------

# f_sample: not implemented
# f_transform
# Note: this function does not actually apply the transform; it just maps x to an orbit representative
function lorentz_transform(x::AbstractVector{Float64}, g::Vector{Float64})
    return g
end
# f_max_inv: not implemented
# f_tau_inv
function tau_inv_lorentz(x::AbstractVector{Float64}; dz=2)
    Q = Vector{Float64}(undef, dz)
    @threads for i in 1:dz
        Q[i] = x[4*(i-1)+1]^2 - sum(x[(4*(i-1)+2):(4*i)].^2)
    end
    return Q
end