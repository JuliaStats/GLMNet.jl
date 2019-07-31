__precompile__()

module GLMNet
using Distributions, StatsBase
using Distributed, Printf, Random, SparseArrays


depsjl = joinpath(@__DIR__, "..", "deps", "deps.jl")
if isfile(depsjl)
    include(depsjl)
else
    error("GLMNet not properly installed. Please run Pkg.build(\"GLMNet\") and restart julia")
end

function __init__()
    check_deps()
end

import Base.getindex, Base.convert, Base.size, Base.show
export glmnet!, glmnet, nactive, predict, glmnetcv, GLMNetPath, GLMNetCrossValidation, CompressedPredictorMatrix

struct CompressedPredictorMatrix <: AbstractMatrix{Float64}
    ni::Int               # Number of predictors
    ca::Matrix{Float64}   # Predictor values
    ia::Vector{Int32}     # Predictor indices
    nin::Vector{Int32}    # Number of predictors in each solution
end

size(X::CompressedPredictorMatrix) = (X.ni, length(X.nin))

function getindex(X::CompressedPredictorMatrix, a::Int, b::Int)
    checkbounds(X, a, b)
    for i = 1:X.nin[b]
        if X.ia[i] == a
            return X.ca[i, b]
        end
    end
    return 0.0
end

function getindex(X::CompressedPredictorMatrix, a::AbstractVector{Int}, b::Int)
    checkbounds(X, a, b)
    out = zeros(length(a))
    for i = 1:X.nin[b]
        if first(a) <= X.ia[i] <= last(a)
            out[X.ia[i] - first(a) + 1] = X.ca[i, b]
        end
    end
    out
end
function getindex(X::CompressedPredictorMatrix, a::AbstractVector{Int}, b::AbstractVector{Int})
    checkbounds(X, a, b)
    out = zeros(length(a), length(b))
    for j = 1:length(b), i = 1:X.nin[b[j]]
        if first(a) <= X.ia[i] <= last(a)
            out[X.ia[i] - first(a) + 1, j] = X.ca[i, b[j]]
        end
    end
    out
end


function getindex(X::CompressedPredictorMatrix, a::Int, b::AbstractVector{Int})
    checkbounds(X, a, b)
    out = zeros(length(b))
    for j = 1:length(b), i = 1:X.nin[b[j]]
        if a == X.ia[i]
            out[j] = X.ca[i, b[j]]
        end
    end
    out
end

# Get number of active predictors for a model in X
# nin can be > non-zero predictors under some circumstances...
function nactive(X::CompressedPredictorMatrix, b::Int)
    n = 0
    for i = 1:X.nin[b]
        n += X.ca[i, b] != 0
    end
    n
end
nactive(X::CompressedPredictorMatrix, b::AbstractVector{Int}=1:length(X.nin)) =
    [nactive(X, j) for j in b]

function convert(::Type{Matrix{Float64}}, X::CompressedPredictorMatrix)
    mat = zeros(X.ni, length(X.nin))
    for b = 1:size(mat, 2), i = 1:X.nin[b]
        mat[X.ia[i], b] = X.ca[i, b]
    end
    return mat
end

function show(io::IO, X::CompressedPredictorMatrix)
    println(io, "$(size(X, 1))x$(size(X, 2)) CompressedPredictorMatrix:")
    Base.showarray(io, convert(Matrix, X); header=false)
end

struct GLMNetPath{F<:Distribution}
    family::F
    a0::Vector{Float64}              # intercept values for each solution
    betas::CompressedPredictorMatrix # coefficient values for each solution
    null_dev::Float64                # Null deviance of the model
    dev_ratio::Vector{Float64}       # R^2 values for each solution
    lambda::Vector{Float64}          # lamda values corresponding to each solution
    npasses::Int                     # actual number of passes over the
                                     # data for all lamda values
end

# Compute the model response to predictors in X
# No inverse link is applied
makepredictmat(path::GLMNetPath, sz::Int, model::Int) = fill(path.a0[model], sz)
makepredictmat(path::GLMNetPath, sz::Int, model::UnitRange{Int}) = repeat(transpose(path.a0[model]), outer=(sz, 1))
function predict(path::GLMNetPath, X::AbstractMatrix,
                 model::Union{Int,AbstractVector{Int}}=1:length(path.a0))
    betas = path.betas
    ca = betas.ca
    ia = betas.ia
    nin = betas.nin

    y = makepredictmat(path, size(X, 1), model)
    for b = 1:length(model)
        m = model[b]
        for i = 1:nin[m]
            iia = ia[i]
            for d = 1:size(X, 1)
                y[d, b] += ca[i, m]*X[d, iia]
            end
        end
    end

    y
end

abstract type Loss end
struct MSE <: Loss
    y::Vector{Float64}
end
loss(l::MSE, i, mu) = abs2(l.y[i] - mu)

struct LogisticDeviance <: Loss
    y::Matrix{Float64}
    fulldev::Vector{Float64}    # Deviance of model with parameter for each y
end
LogisticDeviance(y::Matrix{Float64}) =
    LogisticDeviance(y, [((y[i, 1] == 0.0 ? 0.0 : log(y[i, 1])) +
                          (y[i, 2] == 0.0 ? 0.0 : log(y[i, 2]))) for i = 1:size(y, 1)])

# These are hard-coded in the glmnet Fortran code
const PMIN = 1e-5
const PMAX = 1-1e-5
function loss(l::LogisticDeviance, i, mu)
    expmu = exp(mu)
    lf = expmu/(expmu+1)
    lf = lf < PMIN ? PMIN : lf > PMAX ? PMAX : lf
    2.0*(l.fulldev[i] - (l.y[i, 1]*log1p(-lf) + l.y[i, 2]*log(lf)))
end

struct PoissonDeviance <: Loss
    y::Vector{Float64}
    fulldev::Vector{Float64}    # Deviance of model with parameter for each y
end
PoissonDeviance(y::Vector{Float64}) =
    PoissonDeviance(y, [y == 0.0 ? 0.0 : y*log(y) - y for y in y])
loss(l::PoissonDeviance, i, mu) = 2*(l.fulldev[i] - (l.y[i]*mu - exp(mu)))

devloss(::Normal, y) = MSE(y)
devloss(::Binomial, y) = LogisticDeviance(y)
devloss(::Poisson, y) = PoissonDeviance(y)

# Check the dimensions of X, y, and weights
function validate_x_y_weights(X, y, weights)
    size(X, 1) == size(y, 1) ||
        error(DimensionMismatch("length of y must match rows in X"))
    length(weights) == size(y, 1) ||
        error(DimensionMismatch("length of weights must match y"))
end

# Compute deviance for given model(s) with the predictors in X versus known
# responses in y with the given weight
function loss(path::GLMNetPath, X::AbstractMatrix{Float64},
              y::Union{AbstractVector{Float64}, AbstractMatrix{Float64}},
              weights::AbstractVector{Float64}=ones(size(y, 1)),
              lossfun::Loss=devloss(path.family, y),
              model::Union{Int, AbstractVector{Int}}=1:length(path.a0))
    validate_x_y_weights(X, y, weights)
    mu = predict(path, X, model)
    devs = zeros(size(mu, 2))
    for j = 1:size(mu, 2), i = 1:size(mu, 1)
        devs[j] += loss(lossfun, i, mu[i, j])*weights[i]
    end
    devs/sum(weights)
end
loss(path::GLMNetPath, X::AbstractMatrix, y::Union{AbstractVector, AbstractMatrix},
     weights::AbstractVector=ones(size(y, 1)), va...) =
  loss(path, convert(Matrix{Float64}, X), convert(Array{Float64}, y),
       convert(Vector{Float64}, weights), va...)

modeltype(::Normal) = "Least Squares"
modeltype(::Binomial) = "Logistic"
modeltype(::Multinomial) = "Multinomial"
modeltype(::Poisson) = "Poisson"

function show(io::IO, g::GLMNetPath)
    println(io, "$(modeltype(g.family)) GLMNet Solution Path ($(size(g.betas, 2)) solutions for $(size(g.betas, 1)) predictors in $(g.npasses) passes):")
    print(io, CoefTable(Union{Vector{Int},Vector{Float64}}[nactive(g.betas), g.dev_ratio, g.lambda], ["df", "pct_dev", "λ"], []))
end

function check_jerr(jerr, maxit)
    if 0 < jerr < 7777
        error("glmnet: memory allocation error")
    elseif jerr == 7777
        error("glmnet: all used predictors have zero variance")
    elseif jerr == 1000
        error("glmnet: all predictors are unpenalized")
    elseif -10001 < jerr < 0
        @warn("glment: convergence for $(-jerr)th lambda value not reached after $maxit iterations")
    elseif jerr < -10000
        @warn("glmnet: number of non-zero coefficients along path exceeds $nx at $(maxit+10000)th lambda value")
    end
end

macro validate_and_init()
    esc(quote
        validate_x_y_weights(X, y, weights)
        length(penalty_factor) == size(X, 2) ||
            error(DimensionMismatch("length of penalty_factor must match rows in X"))
        (size(constraints, 1) == 2 && size(constraints, 2) == size(X, 2)) ||
            error(DimensionMismatch("contraints must be a 2 x n matrix"))
        0 <= lambda_min_ratio <= 1 || error("lambda_min_ratio must be in range [0.0, 1.0]")

        if !isempty(lambda)
            # user-specified lambda values
            nlambda == 100 || error("cannot specify both lambda and nlambda")
            lambda_min_ratio == (length(y) < size(X, 2) ? 1e-2 : 1e-4) ||
                error("cannot specify both lambda and lambda_min_ratio")
            nlambda = length(lambda)
            lambda_min_ratio = 2.0
        end

        lmu = Int32[0]
        a0 = zeros(Float64, nlambda)
        ca = Matrix{Float64}(undef, pmax, nlambda)
        ia = Vector{Int32}(undef, pmax)
        nin = Vector{Int32}(undef, nlambda)
        fdev = Vector{Float64}(undef, nlambda)
        alm = Vector{Float64}(undef, nlambda)
        nlp = Int32[0]
        jerr = Int32[0]
    end)
end

macro check_and_return()
    esc(quote
        check_jerr(jerr[1], maxit)

        lmu = lmu[1]
        # first lambda is infinity; changed to entry point
        if isempty(lambda) && length(alm) > 2
            alm[1] = exp(2*log(alm[2])-log(alm[3]))
        end
        X = CompressedPredictorMatrix(size(X, 2), ca[:, 1:lmu], ia, nin[1:lmu])
        GLMNetPath(family, a0[1:lmu], X, null_dev, fdev[1:lmu], alm[1:lmu], Int(nlp[1]))
    end)
end

function glmnet!(X::Matrix{Float64}, y::Vector{Float64},
             family::Normal=Normal();
             weights::Vector{Float64}=ones(length(y)),
             naivealgorithm::Bool=(size(X, 2) >= 500), alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000)
    @validate_and_init

    ccall((:elnet_, libglmnet), Nothing,
          (Ref{Int32}, Ref{Float64}, Ref{Int32}, Ref{Int32}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ref{Int32}, Ptr{Float64}, Ptr{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Int32}, Ref{Float64}, Ptr{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
          (naivealgorithm ? 2 : 1), alpha, size(X, 1), size(X, 2), X, y, weights, 0,
          penalty_factor, constraints, dfmax, pmax, nlambda, lambda_min_ratio, lambda, tol,
          standardize, intercept, maxit, lmu, a0, ca, ia, nin, fdev, alm, nlp, jerr)

    null_dev = 0.0
    mu = mean(y)
    for i = 1:length(y)
        null_dev += abs2(null_dev-mu)
    end

    @check_and_return
end
function glmnet!(X::SparseMatrixCSC{Float64,Int32}, y::Vector{Float64},
             family::Normal=Normal();
             weights::Vector{Float64}=ones(length(y)),
             naivealgorithm::Bool=(size(X, 2) >= 500), alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000)
    @validate_and_init

    ccall((:spelnet_, libglmnet), Nothing,
          (Ref{Int32}, Ref{Float64}, Ref{Int32}, Ref{Int32}, Ptr{Float64}, Ptr{Int32},
           Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ref{Int32}, Ptr{Float64}, Ptr{Float64},
           Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Float64}, Ptr{Float64}, Ref{Float64},
           Ref{Int32}, Ref{Int32}, Ref{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64},
           Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
          (naivealgorithm ? 2 : 1), alpha, size(X, 1), size(X, 2), X.nzval, X.colptr,
          X.rowval, y, weights, 0, penalty_factor, constraints, dfmax, pmax, nlambda,
          lambda_min_ratio, lambda, tol, standardize, intercept, maxit, lmu, a0, ca, ia,
          nin, fdev, alm, nlp, jerr)

    null_dev = 0.0
    mu = mean(y)
    for i = 1:length(y)
        null_dev += abs2(null_dev-mu)
    end

    @check_and_return
end

function glmnet!(X::Matrix{Float64}, y::Matrix{Float64},
             family::Binomial;
             offsets::Union{Vector{Float64},Nothing}=nothing,
             weights::Vector{Float64}=ones(size(y, 1)),
             alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000, algorithm::Symbol=:newtonraphson)
    @validate_and_init
    size(y, 2) == 2 || error("glmnet for logistic models requires a two-column matrix with "*
                             "counts of negative responses in the first column and positive "*
                             "responses in the second")
    kopt = algorithm == :newtonraphson ? 0 :
           algorithm == :modifiednewtonraphson ? 1 :
           algorithm == :nzsame ? 2 : error("unknown algorithm ")
    offsets::Vector{Float64} = isa(offsets, Nothing) ? zeros(size(y, 1)) : copy(offsets)
    length(offsets) == size(y, 1) || error("length of offsets must match length of y")

    null_dev = Vector{Float64}(undef, 1)

    # The Fortran code expects positive responses in first column, but
    # this convention is evidently unacceptable to the authors of the R
    # code, and, apparently, to us
    for i = 1:size(y, 1)
        a = y[i, 1]
        b = y[i, 2]
        y[i, 1] = b*weights[i]
        y[i, 2] = a*weights[i]
    end

    ccall((:lognet_, libglmnet), Nothing,
          (Ref{Float64}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ref{Int32}, Ptr{Float64}, Ptr{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Int32}, Ref{Float64}, Ptr{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Int32}, Ref{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
           Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
          alpha, size(X, 1), size(X, 2), 1, X, y, copy(offsets), 0, penalty_factor,
          constraints, dfmax, pmax, nlambda, lambda_min_ratio, lambda, tol, standardize,
          intercept, maxit, kopt, lmu, a0, ca, ia, nin, null_dev, fdev, alm, nlp, jerr)

    null_dev = null_dev[1]
    @check_and_return
end
function glmnet!(X::SparseMatrixCSC{Float64,Int32}, y::Matrix{Float64},
             family::Binomial;
             offsets::Union{Vector{Float64},Nothing}=nothing,
             weights::Vector{Float64}=ones(size(y, 1)),
             alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000, algorithm::Symbol=:newtonraphson)
    @validate_and_init
    size(y, 2) == 2 || error("glmnet for logistic models requires a two-column matrix with "*
                             "counts of negative responses in the first column and positive "*
                             "responses in the second")
    kopt = algorithm == :newtonraphson ? 0 :
           algorithm == :modifiednewtonraphson ? 1 :
           algorithm == :nzsame ? 2 : error("unknown algorithm ")
    offsets::Vector{Float64} = isa(offsets, Nothing) ? zeros(size(y, 1)) : copy(offsets)
    length(offsets) == size(y, 1) || error("length of offsets must match length of y")

    null_dev = Vector{Float64}(undef, 1)

    # The Fortran code expects positive responses in first column, but
    # this convention is evidently unacceptable to the authors of the R
    # code, and, apparently, to us
    for i = 1:size(y, 1)
        a = y[i, 1]
        b = y[i, 2]
        y[i, 1] = b*weights[i]
        y[i, 2] = a*weights[i]
    end

    ccall((:splognet_, libglmnet), Nothing,
          (Ref{Float64}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ptr{Float64}, Ptr{Int32},
           Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ref{Int32}, Ptr{Float64}, Ptr{Float64},
           Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Float64}, Ptr{Float64}, Ref{Float64},
           Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ptr{Int32}, Ptr{Float64},
           Ptr{Float64}, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64},
           Ptr{Int32}, Ptr{Int32}),
          alpha, size(X, 1), size(X, 2), 1, X.nzval, X.colptr, X.rowval, y, copy(offsets),
          0, penalty_factor, constraints, dfmax, pmax, nlambda, lambda_min_ratio, lambda,
          tol, standardize, intercept, maxit, kopt, lmu, a0, ca, ia, nin, null_dev, fdev,
          alm, nlp, jerr)

    null_dev = null_dev[1]
    @check_and_return
end

function glmnet!(X::Matrix{Float64}, y::Vector{Float64},
             family::Poisson;
             offsets::Union{Vector{Float64},Nothing}=nothing,
             weights::Vector{Float64}=ones(length(y)),
             alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000)
    @validate_and_init
    null_dev = Vector{Float64}(undef, 1)

    offsets::Vector{Float64} = isa(offsets, Nothing) ? zeros(length(y)) : copy(offsets)
    length(offsets) == length(y) || error("length of offsets must match length of y")

    ccall((:fishnet_, libglmnet), Nothing,
          (Ref{Float64}, Ref{Int32}, Ref{Int32}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Float64}, Ref{Int32}, Ptr{Float64}, Ptr{Float64}, Ref{Int32},
           Ref{Int32}, Ref{Int32}, Ref{Float64}, Ptr{Float64}, Ref{Float64}, Ref{Int32},
           Ref{Int32}, Ref{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
           Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
          alpha, size(X, 1), size(X, 2), X, y, offsets, weights, 0, penalty_factor,
          constraints, dfmax, pmax, nlambda, lambda_min_ratio, lambda, tol, standardize,
          intercept, maxit, lmu, a0, ca, ia, nin, null_dev, fdev, alm, nlp, jerr)

    null_dev = null_dev[1]
    @check_and_return
end
function glmnet!(X::SparseMatrixCSC{Float64,Int32}, y::Vector{Float64},
             family::Poisson;
             offsets::Union{Vector{Float64},Nothing}=nothing,
             weights::Vector{Float64}=ones(length(y)),
             alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000)
    @validate_and_init
    null_dev = Vector{Float64}(undef, 1)

    offsets::Vector{Float64} = isa(offsets, Nothing) ? zeros(length(y)) : copy(offsets)
    length(offsets) == length(y) || error("length of offsets must match length of y")

    ccall((:spfishnet_, libglmnet), Nothing,
          (Ref{Float64}, Ref{Int32}, Ref{Int32}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ref{Int32}, Ptr{Float64}, Ptr{Float64},
           Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Float64}, Ptr{Float64}, Ref{Float64},
           Ref{Int32}, Ref{Int32}, Ref{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64},
           Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
           Ptr{Int32}),
          alpha, size(X, 1), size(X, 2), X.nzval, X.colptr, X.rowval, y, offsets, weights,
          0, penalty_factor, constraints, dfmax, pmax, nlambda, lambda_min_ratio, lambda,
          tol, standardize, intercept, maxit, lmu, a0, ca, ia, nin, null_dev, fdev, alm,
          nlp, jerr)

    null_dev = null_dev[1]
    @check_and_return
end

glmnet(X::Matrix{Float64}, y::Vector{Float64}, family::Distribution=Normal(); kw...) =
    glmnet!(copy(X), copy(y), family; kw...)
glmnet(X::AbstractMatrix, y::AbstractVector, family::Distribution=Normal(); kw...) =
    glmnet(convert(Matrix{Float64}, X), convert(Vector{Float64}, y), family; kw...)
glmnet(X::SparseMatrixCSC, y::AbstractVector, family::Distribution=Normal(); kw...) =
    glmnet!(convert(SparseMatrixCSC{Float64,Int32}, X), convert(Vector{Float64}, y), family; kw...)
glmnet(X::Matrix{Float64}, y::Matrix{Float64}, family::Binomial; kw...) =
    glmnet!(copy(X), copy(y), family; kw...)
glmnet(X::SparseMatrixCSC, y::AbstractMatrix, family::Binomial; kw...) =
    glmnet!(convert(SparseMatrixCSC{Float64,Int32}, X), convert(Matrix{Float64}, y), family; kw...)
glmnet(X::Matrix, y::Matrix, family::Binomial; kw...) =
    glmnet(convert(Matrix{Float64}, X), convert(Matrix{Float64}, y), family; kw...)

struct GLMNetCrossValidation
    path::GLMNetPath
    nfolds::Int
    lambda::Vector{Float64}
    meanloss::Vector{Float64}
    stdloss::Vector{Float64}
end

function show(io::IO, cv::GLMNetCrossValidation)
    g = cv.path
    println(io, "$(modeltype(g.family)) GLMNet Cross Validation")
    println(io, "$(length(cv.lambda)) models for $(size(g.betas, 1)) predictors in $(cv.nfolds) folds")
    x, i = findmin(cv.meanloss)
    @printf io "Best λ %.3f (mean loss %.3f, std %.3f)" cv.lambda[i] x cv.stdloss[i]
    print(io, )
end

function glmnetcv(X::AbstractMatrix, y::Union{AbstractVector,AbstractMatrix},
                  family::Distribution=Normal(); weights::Vector{Float64}=ones(length(y)),
                  nfolds::Int=min(10, div(size(y, 1), 3)),
                  folds::Vector{Int}=begin
                      n, r = divrem(size(y, 1), nfolds)
                      shuffle!([repeat(1:nfolds, outer=n); 1:r])
                  end, parallel::Bool=false, kw...)
    # Fit full model once to determine parameters
    X = convert(Matrix{Float64}, X)
    y = convert(Array{Float64}, y)
    path = glmnet(X, y, family; kw...)

    # In case user defined folds
    nfolds = maximum(folds)

    # We shouldn't pass on nlambda and lambda_min_ratio if the user
    # specified these, since that would make us throw errors, and this
    # is entirely determined by the lambda values we will pass
    kw = collect(kw)
    filter!(kw) do akw
        kwname = akw[1]
        kwname != :nlambda && kwname != :lambda_min_ratio && kwname != :lambda
    end

    # Do model fits and compute loss for each
    fits = (parallel ? pmap : map)(1:nfolds) do i
        f = folds .== i
        holdoutidx = findall(f)
        modelidx = findall(!, f)
        g = glmnet!(X[modelidx, :], isa(y, AbstractVector) ? y[modelidx] : y[modelidx, :], family;
                    weights=weights[modelidx], lambda=path.lambda, kw...)
        loss(g, X[holdoutidx, :], isa(y, AbstractVector) ? y[holdoutidx] : y[holdoutidx, :],
             weights[holdoutidx])
    end

    fitloss = hcat(fits...)::Matrix{Float64}

    ninfold = zeros(Int, nfolds)
    for f in folds
        ninfold[f] += 1
    end

    # Mean weighted by fold size
    meanloss = zeros(size(fitloss, 1))
    for j = 1:size(fitloss, 2)
        wfold = ninfold[j]/length(folds)
        for i = 1:size(fitloss, 1)
            meanloss[i] += fitloss[i, j]*wfold
        end
    end

    # Standard deviation weighted by fold size
    stdloss = zeros(size(fitloss, 1))
    for j = 1:size(fitloss, 2)
        wfold = ninfold[j]
        for i = 1:size(fitloss, 1)
            stdloss[i] += abs2(fitloss[i, j] - meanloss[i])*wfold
        end
    end
    for i = 1:size(fitloss, 1)
        stdloss[i] = sqrt(stdloss[i]/length(folds)/(nfolds - 1))
    end

    GLMNetCrossValidation(path, nfolds, path.lambda, meanloss, stdloss)
end
end # module
