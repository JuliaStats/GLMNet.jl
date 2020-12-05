export CoxPH, coef, lambdamin

struct CoxPH <: ContinuousUnivariateDistribution
    theta::Float64 # risk
    CoxPH(theta::Real) = new(Float64(theta))
    CoxPH() = CoxPH(0)
end


function CoxDeviance(risk::Array, y::Matrix,
        weights::AbstractVector{Float64}=ones(size(y, 1))) 
    order = sortperm(y[:, 1])
    y = y[order,:]
    risk = risk[order, :]
    devs = zeros(size(risk, 2))
    for j = 1:size(risk, 2)
        rsk = risk[:, j] .* weights;
        cumrsk = 0;
        for i = size(risk, 1):-1:1
            cumrsk += rsk[i]
            if y[i, 2] == 1
                devs[j] += log(rsk[i]) - log(cumrsk)
            end
        end
    end
    return -devs
end


function predict(path::GLMNetPath{<:CoxPH}, X::AbstractMatrix,
        model::Union{Int,AbstractVector{Int}}=1:length(path.lambda);
        outtype=:link, offsets=zeros(size(X, 1)))
    link = X * path.betas[:, model]
    if any(offsets .!= 0)
        if isa(model, Vector)
            link += repeat(offsets, 1, length(model))
        else
            link += offsets
        end
    end
    if outtype == :link
        return link
    else
        return exp.(link)
    end
end


function loss(path::GLMNetPath{<:CoxPH}, X::AbstractMatrix{Float64},
        y::Union{AbstractVector{Float64},AbstractMatrix{Float64}},
        weights::AbstractVector{Float64}=ones(size(y, 1)),
        model::Union{Int,AbstractVector{Int}}=1:length(path.lambda);
        offsets=zeros(size(X, 1)))
    validate_x_y_weights(X, y, weights)
    risk = exp(predict(path, X, model; offsets=offsets))
    devs = CoxDeviance(risk, y, weights)
    return devs ./ sum(y[:, 2])
end

loss(path::GLMNetPath{<:CoxPH}, X::AbstractMatrix, y::Union{AbstractVector,AbstractMatrix},
        weights::AbstractVector=ones(size(y, 1)), va...; kw...) =
    loss(path, Float64(X), Float64(y), Float64(weights), va...; kw...)

modeltype(::CoxPH) = "Cox's Proportional Model"

function show(io::IO, g::GLMNetPath{<:CoxPH})
    println(io, "$(modeltype(g.family)) GLMNet Solution Path ($(size(g.betas, 2)) solutions for $(size(g.betas, 1)) predictors in $(g.npasses) passes):")
    print(io, DataFrame(df=nactive(g.betas), pct_dev=g.dev_ratio, λ=g.lambda))
end


macro check_and_return_cox()
    esc(quote
        check_jerr(jerr[1], maxit,pmax)
        lmu = lmu[1]
        # first lambda is infinity; changed to entry point
        if isempty(lambda) && length(alm) > 2
            alm[1] = exp(2*log(alm[2])-log(alm[3]))
        end
        X = CompressedPredictorMatrix(size(X, 2), ca[:, 1:lmu], ia, nin[1:lmu])
        GLMNetPath(family, a0[1:lmu], X, null_dev[1], fdev[1:lmu], alm[1:lmu], Int(nlp[1]))
    end)
end


function glmnet!(X::Matrix{Float64}, y::Matrix{Float64}, family::CoxPH;
             offsets::Vector{Float64}=zeros(size(y,1)),
             weights::Vector{Float64}=ones(size(y,1)),
             alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2)+1, pmax::Int=min(dfmax*2+20, size(X, 2)), 
             nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000)
    @validate_and_init
    @assert size(y) == tuple(size(y, 1), 2)
    times  = y[:, 1]
    status = y[:, 2]
    nobs = Int32(size(X, 1))
    nvars = Int32(size(X, 2))
    dfmax = Int32(dfmax);
    pmax = Int32(pmax);
    nlambda = Int32(nlambda);
    lambda_min_ratio = float(lambda_min_ratio);
    lambda = convert(Vector{Float64}, lambda);
    tol = float(tol);
    standardize = Int32(standardize);
    intercept = Int32(intercept);
    maxit = Int32(maxit);
    null_dev = [0.0];
    jd = Int32(0);
    #
    ca = zeros(Float64, pmax, nlambda) # fitted coef/param
    ia = zeros(Int32, pmax) # param order
    nin = zeros(Int32, nlambda) # 
    fdev = zeros(Float64, nlambda)
    alm = zeros(Float64, nlambda)
    #
    length(offsets) == length(times) || error("length of offsets must match length of y")
    offsets = copy(offsets)
    #
    ccall((:coxnet_, libglmnet), Nothing, (
        Ptr{Float64} , Ptr{Int32}     , Ptr{Int32}        , Ptr{Float64} , # 1
        Ptr{Float64} , Ptr{Float64}   , Ptr{Float64}      , Ptr{Float64} , # 2
        Ptr{Int32}   , Ptr{Float64}   , Ptr{Float64}      , Ptr{Int32}   , # 3
        Ptr{Int32}   , Ptr{Int32}     , Ptr{Float64}      , Ptr{Float64} , # 4
        Ptr{Float64} , Ptr{Int32}     , Ptr{Int32}        , Ptr{Int32}   , # 5
        Ptr{Float64} , Ptr{Int32}     , Ptr{Int32}        , Ptr{Float64} , # 6
        Ptr{Float64} , Ptr{Float64}   , Ptr{Int32}        , Ptr{Int32}     # 7
        ),
        Ref(alpha)       , Ref(nobs)          , Ref(nvars)            , X            , # 1
        times        , status         , offsets           , weights      , # 2
        Ref(jd)          , penalty_factor , constraints       , Ref(dfmax)       , # 3
        Ref(pmax)        , Ref(nlambda)       , Ref(lambda_min_ratio) , lambda       , # 4
        Ref(tol)         , Ref(maxit)         , Ref(standardize)      , lmu          , # 5
        ca           , ia             , nin               , null_dev     , # 6
        fdev         , alm            , nlp               , jerr           # 7
        )
    @check_and_return_cox
end


glmnet(X::Matrix{Float64}, y::Matrix{Float64}, family::CoxPH; kw...) =
    glmnet!(copy(X), copy(y), family; kw...)

glmnet(X::AbstractMatrix, y::AbstractMatrix, family::CoxPH; kw...) =
    glmnet(convert(Matrix{Float64}, X), convert(Matrix{Float64}, y), family; kw...)

function glmnet(X::AbstractMatrix, time::AbstractVector,
    status::AbstractVector, family::CoxPH = CoxPH(); kw...)
    #
    @assert size(X, 1) == length(time) == length(status)
    y = [time status]
    glmnet(X, y, family; kw...)
end


function glmnetcv(X::AbstractMatrix, time::Vector, status::Vector, family = CoxPH(); kw...)
    @assert size(X, 1) == length(time) == length(status)
    y = [time status]
    glmnetcv(X, y, CoxPH(); kw...)
end


function predict(pathcv::GLMNetCrossValidation, X::AbstractMatrix; outtype = :link, kw...)
    ind = argmin(pathcv.meanloss)
    predict(pathcv.path, X, ind; outtype = outtype, kw...)
end

lambdamin(pathcv::GLMNetCrossValidation) = pathcv.lambda[argmin(pathcv.meanloss)]

function coef(pathcv::GLMNetCrossValidation)
    ind = argmin(pathcv.meanloss)
    if isa(pathcv.path.family, Multinomial)
        pathcv.path.betas[:,:,ind]
    else
        pathcv.path.betas[:, ind]
    end
end
