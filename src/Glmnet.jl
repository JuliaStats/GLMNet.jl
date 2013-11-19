module Glmnet
using DataFrames

const libglmnet = joinpath(Pkg.dir("Glmnet"), "deps", "libglmnet.so")

import Base.getindex, Base.convert, Base.size, Base.show
export fit!, fit

immutable CompressedPredictorMatrix <: AbstractMatrix{Float64}
    ni::Int
    ca::Matrix{Float64}
    ia::Vector{Int32}
    df::Vector{Int32}
end

size(X::CompressedPredictorMatrix) = (X.ni, length(X.df))

function getindex(X::CompressedPredictorMatrix, a::Int, b::Int)
    a <= X.ni && b <= length(X.df) || throw(BoundsError())
    for i = 1:X.df[b]
        if X.ia[i] == a
            return X.ca[i, b]
        end
    end
    return 0.0
end

function getindex(X::CompressedPredictorMatrix, a::AbstractVector{Int}, b::Int)
    checkbounds(X, a, b)
    out = zeros(length(a))
    for i = 1:X.df[b]
        if first(a) <= X.ia[i] <= last(a)
            out[X.ia[i] - first(a) + 1] = X.ca[i, b]
        end
    end
    out
end

function getindex(X::CompressedPredictorMatrix, a::Union(Int, AbstractVector{Int}), b::AbstractVector{Int})
    checkbounds(X, a, b)
    out = zeros(length(a), length(b))
    for j = 1:length(b), i = 1:X.df[b[j]]
        if first(a) <= X.ia[i] <= last(a)
            out[X.ia[i] - first(a) + 1, j] = X.ca[i, b[j]]
        end
    end
    out
end

function convert{T<:Matrix{Float64}}(::Type{T}, X::CompressedPredictorMatrix)
    mat = zeros(X.ni, length(X.df))
    for b = 1:size(mat, 2), i = 1:X.df[b]
        mat[X.ia[i], b] = X.ca[i, b]
    end
    return mat
end

function show(io::IO, X::CompressedPredictorMatrix)
    println(io, "$(size(X, 1))x$(size(X, 2)) CompressedPredictorMatrix:")
    Base.showarray(io, convert(Matrix, X); header=false)
end

immutable GlmnetPath
    a0::Vector{Float64}             # intercept values for each solution
    betas::CompressedPredictorMatrix    # coefficient values for each solution
    dev_ratio::Vector{Float64}      # R^2 values for each solution
    λ::Vector{Float64}              # lamda values corresponding to each solution
    npasses::Int                    # actual number of passes over the
                                    # data for all lamda values
end

function show(io::IO, g::GlmnetPath)
    println(io, "Glmnet Solution Path ($(size(g.betas, 2)) solutions for $(size(g.betas, 1)) predictors in $(g.npasses) passes):")
    print(DataFrame({g.betas.df, g.dev_ratio, g.λ}, ["df", "%dev", "λ"]))
end

function fit!(X::StridedMatrix{Float64}, y::StridedVector{Float64},
             weights::StridedVector{Float64}=ones(length(y));
             naivealgorithm::Bool=(size(X, 2) >= 500), alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000)
    size(X, 1) == length(y) ||
        error(Base.LinAlg.DimensionMismatch("length of y must match rows in X"))
    length(weights) == length(y) ||
        error(Base.LinAlg.DimensionMismatch("length of weights must match y"))
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
    rsq = Array(Float64, nlambda)
    alm = Array(Float64, nlambda)
    nlp = Int32[0]
    jerr = Int32[0]

    ccall((:elnet_, libglmnet), Void,
          (Ptr{Int32}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Ptr{Int32}, Ptr{Int32}, Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
           Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
          &(naivealgorithm ? 2 : 1), &alpha, &size(X, 1), &size(X, 2), X, y, weights, &0, penalty_factor,
          constraints, &dfmax, &pmax, &nlambda, &lambda_min_ratio, lambda, &tol, &standardize,
          &intercept, &maxit, lmu, a0, ca, ia, nin, rsq, alm, nlp, jerr)

    jerr = jerr[1]
    if 0 < jerr < 7777
        error("glmnet: memory allocation error")
    elseif jerr == 7777
        error("glmnet: all used predictors have zero variance")
    elseif jerr == 1000
        error("glmnet: all predictors are unpenalized")
    elseif -10001 < jerr < 0
        warn("glment: convergence for $(jerr)th lambda value not reached after $maxit iterations")
    elseif jerr < -10000
        warn("glmnet: number of non-zero coefficients along path exceeds $nx at $(maxit+10000)th lambda value")
    end

    lmu = lmu[1]
    # first lambda is infinity; changed to entry point
    if isempty(lambda) && length(alm) > 2
        alm[1] = exp(2*log(alm[2])-log(alm[3]))
    end
    X = CompressedPredictorMatrix(size(X, 2), ca[:, 1:lmu], ia, nin[1:lmu])
    GlmnetPath(a0[1:lmu], X, rsq[1:lmu], alm[1:lmu], int(nlp[1]))
end

fit(X::StridedMatrix{Float64}, y::StridedVector{Float64}; kw...) = fit!(X, copy(y); kw...)
fit(X::StridedMatrix{Float64}, y::StridedVector{Float64},
    weights::StridedVector{Float64}; kw...) = fit!(X, copy(y), copy(weights); kw...)
fit(X::StridedMatrix, y::StridedVector; kw...) = fit(float64(X), float64(y))
end # module
