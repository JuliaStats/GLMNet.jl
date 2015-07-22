immutable CoxPH <: ContinuousUnivariateDistribution
    theta::Float64 # risk
    CoxPH(theta::Real) = new(float64(theta))
    CoxPH() = new(float64(0))
end


immutable CoxNetPath
    family::CoxPH
    a0::Vector{Float64}              # intercept values for each solution
    betas::CompressedPredictorMatrix # coefficient values for each solution
    null_dev::Float64                # Null deviance of the model
    dev_ratio::Vector{Float64}       # R^2 values for each solution
    lambda::Vector{Float64}          # lamda values corresponding to each solution
    npasses::Int                     # actual number of passes over the
                                     # data for all lamda values
end

function coxDeviance(risk::Array, y::Matrix{Float64}, weights::AbstractVector{Float64}=ones(size(y, 1))) 
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
    return devs
end
    

function loss(path::CoxNetPath, X::AbstractMatrix{Float64},
        y::Union(AbstractVector{Float64}, AbstractMatrix{Float64}),
        weights::AbstractVector{Float64}=ones(size(y, 1)),
        model::Union(Int, AbstractVector{Int})=1:length(path.a0))
    validate_x_y_weights(X, y, weights)
    risk = exp(predict(path, X, model))
    devs = coxDeviance(risk, y, weights)
    return -devs/sum(y[:, 2])
end

loss(path::CoxNetPath, X::AbstractMatrix, y::Union(AbstractVector, AbstractMatrix),
        weights::AbstractVector=ones(size(y, 1)), va...) = 
	loss(path, float64(X), float64(y), float64(weights), va...)


modeltype(::CoxPH) = "Cox's Proportional Model"


macro check_and_return_cox()
    esc(quote
        check_jerr(jerr[1], maxit)
        lmu = lmu[1]
        # first lambda is infinity; changed to entry point
        if isempty(lambda) && length(alm) > 2
            alm[1] = exp(2*log(alm[2])-log(alm[3]))
        end
        X = CompressedPredictorMatrix(size(X, 2), ca[:, 1:lmu], ia, nin[1:lmu])
        CoxNetPath(family, a0[1:lmu], X, null_dev, fdev[1:lmu], alm[1:lmu], int(nlp[1]))
    end)
end


function glmnet!(X::Matrix{Float64}, y::Matrix, family::CoxPH;
             offsets::Union(Vector{Float64}, Nothing)=nothing,
             weights::Vector{Float64}=ones(length(y)),
             alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2)+1, pmax::Int=min(dfmax*2+20, size(X, 2)), 
             nlambda::Int=100,
             lambda_min_ratio::Real=(length(y) < size(X, 2) ? 1e-2 : 1e-4),
             lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
             intercept::Bool=true, maxit::Int=1000000)
    @validate_and_init
    assert(size(y) == tuple(size(y, 1), 2))
    times  = convert(Vector{Float64}, y[:, 1])
    status = convert(Vector{Float64}, y[:, 2])
    nobs = int32(size(X, 1))
    nvars = int32(size(X, 2))
    dfmax = int32(dfmax);
    pmax = int32(pmax);
    nlambda = int32(nlambda);
    lambda_min_ratio = float(lambda_min_ratio);
    lambda = convert(Vector{Float64}, lambda);
    tol = float(tol);
    standardize = int32(standardize);
    intercept = int32(intercept);
    maxit = int32(maxit);
    null_dev = 0.0;
    jd = int32(0);
    #
    ca = zeros(Float64, pmax, nlambda) # fitted coef/param
    ia = zeros(Int32, pmax) # param order
    nin = zeros(Int32, nlambda) # 
    fdev = zeros(Float64, nlambda)
    alm = zeros(Float64, nlambda)
    #
    offsets = isa(offsets, Nothing) ? zeros(length(times)) : copy(offsets)
    length(offsets) == length(times) || error("length of offsets must match length of y")
    #
    ccall((:coxnet_, libglmnet), Void, (
        Ptr{Float64} , Ptr{Int32}     , Ptr{Int32}        , Ptr{Float64} , # 1
        Ptr{Float64} , Ptr{Float64}   , Ptr{Float64}      , Ptr{Float64} , # 2
        Ptr{Int32}   , Ptr{Float64}   , Ptr{Float64}      , Ptr{Int32}   , # 3
        Ptr{Int32}   , Ptr{Int32}     , Ptr{Float64}      , Ptr{Float64} , # 4
        Ptr{Float64} , Ptr{Int32}     , Ptr{Int32}        , Ptr{Int32}   , # 5
        Ptr{Float64} , Ptr{Int32}     , Ptr{Int32}        , Ptr{Float64} , # 6
        Ptr{Float64} , Ptr{Float64}   , Ptr{Int32}        , Ptr{Int32}     # 7
        ),
        &alpha       , &nobs          , &nvars            , X            , # 1
        times        , status         , offsets           , weights      , # 2
        &jd          , penalty_factor , constraints       , &dfmax       , # 3
        &pmax        , &nlambda       , &lambda_min_ratio , lambda       , # 4
        &tol         , &maxit         , &standardize      , lmu          , # 5
        ca           , ia             , nin               , &null_dev    , # 6
        fdev         , alm            , nlp               , jerr           # 7
        )
    @check_and_return_cox
end


glmnet(X::Matrix{Float64}, y::Matrix, family::CoxPH; kw...) =
    glmnet!(copy(X), copy(y), family; kw...)


function glmnet(X::Matrix{Float64}, time::Vector{Float64}, 
	status::Vector{Int}, family::CoxPH = CoxPH(); kw...)
	#
	assert(size(x, 1) == length(time) == length(status))
	y = [time status]
    glmnet!(copy(X), y, family; kw...)
end



predict(path::CoxNetPath, X::AbstractMatrix, args...) = X * path.betas


function glmnetcv(X::AbstractMatrix, y::AbstractMatrix,
            family::CoxPH; weights::Vector{Float64}=ones(size(X,1)),
            grouped = true,
            nfolds::Int=min(10, div(size(y, 1), 3)),
            folds::Vector{Int}=begin
                n, r = divrem(size(y, 1), nfolds)
                shuffle!([repmat(1:nfolds, n); 1:r])
            end, parallel::Bool=false, kw...)
    # Fit full model once to determine parameters
    X = convert(Matrix{Float64}, X)
    y = convert(Matrix{Float64}, y)
    path = glmnet(X, y, family; kw...)

    # In case user defined folds
    nfolds = maximum(folds)

    # We shouldn't pass on nlambda and lambda_min_ratio if the user
    # specified these, since that would make us throw errors, and this
    # is entirely determined by the lambda values we will pass
    filter!(kw) do akw
        kwname = akw[1]
        kwname != :nlambda && kwname != :lambda_min_ratio && kwname != :lambda
    end

    # Do model fits and compute loss for each
    fits = (parallel ? pmap : map)(1:nfolds) do i
        f = folds .== i
        holdoutidx = find(f)
        modelidx = find(!f)
        g = glmnet!(X[modelidx, :], isa(y, AbstractVector) ? y[modelidx] : y[modelidx, :], family;
                    weights=weights[modelidx], lambda=path.lambda, kw...)
        #
        risks = exp(predict(g, X) + repmat(offsets, 1, length(path.lambda)))
        if grouped
            plfull = coxDeviance(risks, y, weights)
            plminusk = coxDeviance(risks[modelidx,:], y[modelidx,:], weights[modelidx])
            plfull - plminusk
        else
            coxDeviance(risks[holdoutidx, :], y[holdoutidx, :], weights[holdoutidx])
        end
    end

    fitloss = -hcat(fits...)::Matrix{Float64}
    meanloss = mean(fitloss, 2)
    stdloss = std(fitloss, 2)

    GLMNetCrossValidation(path, nfolds, path.lambda, meanloss, stdloss)
end
