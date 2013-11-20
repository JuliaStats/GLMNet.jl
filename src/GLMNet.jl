module GLMNet
using DataFrames, Distributions

const libglmnet = joinpath(Pkg.dir("GLMNet"), "deps", "libglmnet.so")

import Base.getindex, Base.convert, Base.size, Base.show, Distributions.fit
export fit!, fit, df

immutable CompressedPredictorMatrix <: AbstractMatrix{Float64}
    ni::Int
    ca::Matrix{Float64}
    ia::Vector{Int32}
    nin::Vector{Int32}
end

size(X::CompressedPredictorMatrix) = (X.ni, length(X.nin))

function getindex(X::CompressedPredictorMatrix, a::Int, b::Int)
    a <= X.ni && b <= length(X.nin) || throw(BoundsError())
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

function getindex(X::CompressedPredictorMatrix, a::Union(Int, AbstractVector{Int}), b::AbstractVector{Int})
    checkbounds(X, a, b)
    out = zeros(length(a), length(b))
    for j = 1:length(b), i = 1:X.nin[b[j]]
        if first(a) <= X.ia[i] <= last(a)
            out[X.ia[i] - first(a) + 1, j] = X.ca[i, b[j]]
        end
    end
    out
end

# nin can be > non-zero predictors under some circumstances...
function df(X::CompressedPredictorMatrix)
    [begin
        n = 0
        for i = 1:X.nin[j]
            n += X.ca[i, j] != 0
        end
        n
    end for j = 1:length(X.nin)]
end

function convert{T<:Matrix{Float64}}(::Type{T}, X::CompressedPredictorMatrix)
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

immutable GLMNetPath
    family::Distribution
    a0::Vector{Float64}              # intercept values for each solution
    betas::CompressedPredictorMatrix # coefficient values for each solution
    null_dev::Float64                # Null deviance of the model
    dev_ratio::Vector{Float64}       # R^2 values for each solution
    λ::Vector{Float64}               # lamda values corresponding to each solution
    npasses::Int                     # actual number of passes over the
                                     # data for all lamda values
end

modeltype(::Normal) = "Least Squares"
modeltype(::Binomial) = "Logistic"
modeltype(::Multinomial) = "Multinomial"
modeltype(::Poisson) = "Poisson"

function show(io::IO, g::GLMNetPath)
    println(io, "$(modeltype(g.family)) GLMNet Solution Path ($(size(g.betas, 2)) solutions for $(size(g.betas, 1)) predictors in $(g.npasses) passes):")
    print(DataFrame({df(g.betas), g.dev_ratio, g.λ}, ["df", "%dev", "λ"]))
end

function check_jerr(jerr, maxit)
    if 0 < jerr < 7777
        error("glmnet: memory allocation error")
    elseif jerr == 7777
        error("glmnet: all used predictors have zero variance")
    elseif jerr == 1000
        error("glmnet: all predictors are unpenalized")
    elseif -10001 < jerr < 0
        warn("glment: convergence for $(-jerr)th lambda value not reached after $maxit iterations")
    elseif jerr < -10000
        warn("glmnet: number of non-zero coefficients along path exceeds $nx at $(maxit+10000)th lambda value")
    end
end

macro validate_and_init()
    esc(quote
        size(X, 1) == size(y, 1) ||
            error(Base.LinAlg.DimensionMismatch("length of y must match rows in X"))
        length(penalty_factor) == size(X, 2) ||
            error(Base.LinAlg.DimensionMismatch("length of penalty_factor must match rows in X"))
        (size(constraints, 1) == 2 && size(constraints, 2) == size(X, 2)) ||
            error(Base.LinAlg.DimensionMismatch("contraints must be a 2 x n matrix"))
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
        a0 = Array(Float64, nlambda)
        ca = Array(Float64, pmax, nlambda)
        ia = Array(Int32, pmax)
        nin = Array(Int32, nlambda)
        fdev = Array(Float64, nlambda)
        alm = Array(Float64, nlambda)
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
        GLMNetPath(family, a0[1:lmu], X, null_dev, fdev[1:lmu], alm[1:lmu], int(nlp[1]))
    end)
end

function fit!(X::StridedMatrix{Float64}, y::StridedVector{Float64},
             family::Normal=Normal();
             weights::StridedVector{Float64}=ones(length(y)),
             naivealgorithm::Bool=(size(X, 2) >= 500), alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000)
    @validate_and_init
    length(weights) == size(y, 1) ||
        error(Base.LinAlg.DimensionMismatch("length of weights must match y"))

    ccall((:elnet_, libglmnet), Void,
          (Ptr{Int32}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
          &(naivealgorithm ? 2 : 1), &alpha, &size(X, 1), &size(X, 2), X, y, weights, &0,
          penalty_factor, constraints, &dfmax, &pmax, &nlambda, &lambda_min_ratio, lambda, &tol,
          &standardize, &intercept, &maxit, lmu, a0, ca, ia, nin, fdev, alm, nlp, jerr)

    null_dev = 0.0
    mu = mean(y)
    for i = 1:length(y)
        null_dev += abs2(null_dev-mu)
    end

    @check_and_return
end

function fit!(X::StridedMatrix{Float64}, y::StridedMatrix{Float64},
             family::Binomial;
             offsets::StridedVector{Float64}=zeros(length(y)),
             alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000, algorithm::Symbol=:newtonraphson)
    @validate_and_init
    size(y, 2) == 2 || error("fit! for logistic models requires a two-column matrix with counts "*
                             "of positive responses in the first column and negative responses "*
                             "in the second")
    kopt = algorithm == :newtonraphson ? 0 :
           algorithm == :modifiednewtonraphson ? 1 :
           algorithm == :nzsame ? 2 : error("unknown algorithm ")

    null_dev = Array(Float64, 1)

    ccall((:lognet_, libglmnet), Void,
          (Ptr{Float64}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64},  Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
           Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
           Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
           Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
          &alpha, &size(X, 1), &size(X, 2), &1, X, y, offsets, &0, penalty_factor,
          constraints, &dfmax, &pmax, &nlambda, &lambda_min_ratio, lambda, &tol, &standardize,
          &intercept, &maxit, &kopt, lmu, a0, ca, ia, nin, null_dev, fdev, alm, nlp, jerr)

    null_dev = null_dev[1]
    @check_and_return
end

function fit!(X::StridedMatrix{Float64}, y::StridedVector{Float64},
             family::Poisson;
             offsets::StridedVector{Float64}=zeros(length(y)),
             weights::StridedVector{Float64}=ones(length(y)),
             alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000)
    @validate_and_init
    null_dev = Array(Float64, 1)

    ccall((:fishnet_, libglmnet), Void,
          (Ptr{Float64}, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
           Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
           Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32},
           Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
          &alpha, &size(X, 1), &size(X, 2), X, y, offsets, weights, &0, penalty_factor,
          constraints, &dfmax, &pmax, &nlambda, &lambda_min_ratio, lambda, &tol, &standardize,
          &intercept, &maxit, lmu, a0, ca, ia, nin, null_dev, fdev, alm, nlp, jerr)

    null_dev = null_dev[1]
    @check_and_return
end

fit(X::StridedMatrix{Float64}, y::StridedVector{Float64}, family=Normal(); kw...) =
    fit!(X, copy(y), family; kw...)
fit(X::StridedMatrix, y::StridedVector, family=Normal(); kw...) =
    fit(float64(X), float64(y), family)
function fit(X::StridedMatrix{Float64}, y::StridedMatrix{Float64}, family::Binomial; kw...)
    size(y, 2) == 2 || error("fit for logistic models requires a two-column matrix with counts "*
                             "of negative responses in the first column and positive responses "*
                             "in the second")
    fit!(X, fliplr(y), family)
end
fit(X::StridedMatrix, y::StridedMatrix, family::Binomial; kw...) =
    fit(float64(X), float64(y), family)
end # module
