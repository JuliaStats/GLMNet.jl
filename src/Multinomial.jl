import Distributions.Multinomial

Multinomial() = Multinomial(1, 1)
modeltype(::Multinomial) = "Multinomial"

function locSoftmax(xs)
    locMax = maximum(xs)
    expPart = exp.(xs .- locMax)
    expPart ./ sum(expPart)
end

function predict(path::GLMNetPath{<:Multinomial}, X::AbstractMatrix,
         model::Union{Int, AbstractVector{Int}}=1:length(path.lambda); 
         outtype = :link, offsets = zeros(size(X, 1), size(path.betas, 2)))
    nresp = size(path.betas, 2);
    out = zeros(Float64, size(X, 1), nresp, length(model));
    for i = 1:length(model)
        out[:, :, i] = repeat(path.a0[:,model[i]]', size(X, 1)) + X * path.betas[:, :, model[i]] + offsets
    end
    if outtype != :link
        for i = 1:size(X, 1), j = 1:length(model)
            out[i, :, j] = locSoftmax(out[i, :, j])
        end
    end
    if length(model) == 1
        return out[:, :, 1]
    else
        return out
    end
end


function MultinomialDeviance(y::Matrix{Float64}, p::Matrix{Float64}, 
    weights::AbstractVector{Float64}=ones(size(y, 1)))
    @assert size(p) == size(y)
    @assert size(p,1) == length(weights)
    p = map(p) do x # round p to be within [PMIN, PMAX]
        x < PMIN ? PMIN :
            x > PMAX ? PMAX : x
    end
    -2*sum(y .* log.(p) .* repeat(weights, 1, size(y, 2))) / sum(weights)
end


function loss(path::GLMNetPath{<:Multinomial}, X::AbstractMatrix{Float64},
              y::Union{AbstractVector{Float64}, AbstractMatrix{Float64}},
              weights::AbstractVector{Float64}=ones(size(y, 1)),
              model::Union{Int, AbstractVector{Int}}=1:length(path.lambda);
              offsets = zeros(size(X,1), size(path.betas, 2)))
    validate_x_y_weights(X, y, weights)
    prob = predict(path, X, model; outtype = :prob, offsets = offsets)
    convert(Vector{Float64}, [MultinomialDeviance(y, prob[:,:, i], weights) for i in 1:length(model)])
end


loss(path::GLMNetPath{<:Multinomial}, X::AbstractMatrix, y::Union{AbstractVector, AbstractMatrix}, 
        weights::AbstractVector=ones(size(y, 1)), va...; kw...) =
    loss(path, convert(Matrix{Float64}, X), 
        convert(Array{Float64}, y),
        convert(Vector{Float64}, weights), va...; kw...)


# Get number of active predictors for a model in X
# nin can be > non-zero predictors under some circumstances...
nactive(X::Array{Float64, 3}, b::Int) = sum(sum(X[:,:,b] .!= 0., dims=1) .> 0)

nactive(X::Array{Float64, 3}, b::AbstractVector{Int}=1:size(X, 3)) =
    [nactive(X, j) for j in b]


function show(io::IO, g::GLMNetPath{<:Multinomial})
    println(io, "$(modeltype(g.family)) GLMNet Solution Path ($(size(g.betas, 2)) solutions for $(size(g.betas, 1)) predictors in $(g.npasses) passes):")
    print(io, DataFrame(df=nactive(g.betas), pct_dev=g.dev_ratio, λ=g.lambda))
end


macro validate_and_init_multi()
    esc(quote
        validate_x_y_weights(X, y, weights)
        length(penalty_factor) == size(X, 2) ||
            error(Base.LinAlg.DimensionMismatch("length of penalty_factor must match rows in X"))
        (size(constraints, 1) == 2 && size(constraints, 2) == size(X, 2)) ||
            error(Base.LinAlg.DimensionMismatch("contraints must be a 2 x n matrix"))
        0 <= lambda_min_ratio <= 1 || error("lambda_min_ratio must be in range [0.0, 1.0]")
        #
        if !isempty(lambda)
            # user-specified lambda values
            nlambda == 100 || error("cannot specify both lambda and nlambda")
            lambda_min_ratio == (length(y) < size(X, 2) ? 1e-2 : 1e-4) ||
                error("cannot specify both lambda and lambda_min_ratio")
            nlambda = length(lambda)
            lambda_min_ratio = 2.0
        end
        #
        alpha = float(alpha)
        nobs = Int32(size(X, 1))
        nvars = Int32(size(X, 2))
        nresp = Int32(size(y, 2))
        dfmax = Int32(dfmax)
        pmax = Int32(pmax)
        nlambda = Int32(nlambda);
        lambda_min_ratio = float(lambda_min_ratio)
        lambda = convert(Vector{Float64}, lambda)
        tol = float(tol)
        standardize = Int32(standardize)
        intercept = Int32(intercept)
        maxit = Int32(maxit)
        null_dev = [0.0]
        jd = Int32(0)
        #
        lmu = Int32[0]
        a0 = zeros(Float64, nresp, nlambda)
        ca = zeros(Float64, pmax, nresp, nlambda)
        ia = zeros(Int32, pmax)
        nin = zeros(Int32, nlambda)
        fdev = zeros(Float64, nlambda)
        alm = zeros(Float64, nlambda)
        nlp = Int32[0]
        jerr = Int32[0]
    end)
end


macro check_and_return_multi()
    esc(quote
        check_jerr(jerr[1], maxit,pmax)
        lmu = lmu[1]
        # first lambda is infinity; changed to entry point
        if isempty(lambda) && length(alm) > 2
            alm[1] = exp(2*log(alm[2])-log(alm[3]))
        end
        a0 = a0 .- repeat(mean(a0, dims=1), size(a0, 1))
        GLMNetPath(family, a0[:, 1:lmu], ca[sortperm(ia), :, 1:lmu], 
            null_dev[1], fdev[1:lmu], alm[1:lmu], Int(nlp[1]))
    end)
end


function glmnet!(X::Matrix{Float64}, y::Matrix{Float64},
             family::Multinomial;
             offsets::Matrix{Float64}=y*0.,
             weights::Vector{Float64}=ones(size(X, 1)),
             alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[a for a in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2)+1, pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000, grouped_multinomial::Bool=false,  
             algorithm::Symbol=:newtonraphson)
    @validate_and_init_multi
    kopt = grouped_multinomial ? Int32(2) : 
        algorithm == :newtonraphson ? Int32(0) :
        algorithm == :modifiednewtonraphson ? Int32(1) : 
        algorithm == :nzsame ? Int32(2) : 
        error("unknown algorithm ")
    # check offsets
    @assert size(y) == size(offsets)
    offsets = copy(offsets)
    y = y .* repeat(weights, 1, size(y, 2))

    ccall(
        (:lognet_, libglmnet), Nothing, (
            Ptr{Float64}   , Ptr{Int32}        , Ptr{Int32}   , Ptr{Int32}   , # 1
            Ptr{Float64}   , Ptr{Float64}      , Ptr{Float64} , Ptr{Int32}   , # 2
            Ptr{Float64}   , Ptr{Float64}      , Ptr{Int32}   , Ptr{Int32}   , # 3
            Ptr{Int32}     , Ptr{Float64}      , Ptr{Float64} , Ptr{Float64} , # 4
            Ptr{Int32}     , Ptr{Int32}        , Ptr{Int32}   , Ptr{Int32}   , # 5
            Ptr{Int32}     , Ptr{Float64}      , Ptr{Float64} , Ptr{Int32}   , # 6
            Ptr{Int32}     , Ptr{Float64}      , Ptr{Float64} , Ptr{Float64} , # 7
            Ptr{Int32}     , Ptr{Int32}                                        # 8
            ),
            Ref(alpha)         , Ref(nobs)             , Ref(nvars)       , Ref(nresp)       , # 1
            X              , y                 , offsets      , Ref(jd)          , # 2
            penalty_factor , constraints       , Ref(dfmax)       , Ref(pmax)        , # 3
            Ref(nlambda)       , Ref(lambda_min_ratio) , lambda       , Ref(tol)         , # 4
            Ref(standardize)   , Ref(intercept)        , Ref(maxit)       , Ref(kopt)        , # 5
            lmu            , a0                , ca           , ia           , # 6
            nin            , null_dev          , fdev         , alm          , # 7
            nlp            , jerr                                              # 8
        )
    @check_and_return_multi
end

glmnet(X::Matrix{Float64}, y::Matrix{Float64}, family::Multinomial; kw...) =
    glmnet!(copy(X), copy(y), family; kw...)

glmnet(X::AbstractMatrix, y::AbstractMatrix, family::Multinomial; kw...) =
    glmnet(convert(Matrix{Float64}, X), convert(Matrix{Float64}, y), family; kw...)

function glmnet(X::Matrix{Float64}, y; kw...)
    lev = sort(unique(y))
    if length(lev) >= 2
        y = convert(Matrix{Float64}, [i == j for i in y, j in lev])
        if length(lev) == 2
            glmnet(X, y, Binomial(); kw...)
        else 
            glmnet(X, y, Multinomial(); kw...)
        end
    else 
        error("y has only one level.")
    end
end

function glmnetcv(X::Matrix{Float64}, y; kw...)
    lev = sort(unique(y))
    if length(lev) >= 2
        y = convert(Matrix{Float64}, [i == j for i in y, j in lev])
        if length(lev) == 2
            glmnetcv(X, y, Binomial(); kw...)
        else
            glmnetcv(X, y, Multinomial(); kw...)
        end
    else
        error("y has only one level.")
    end
end
