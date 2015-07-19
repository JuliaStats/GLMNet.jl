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

function loss(path::CoxNetPath, X::AbstractMatrix{Float64},
              y::Union(AbstractVector{Float64}, AbstractMatrix{Float64}),
              weights::AbstractVector{Float64}=ones(size(y, 1)),
              model::Union(Int, AbstractVector{Int})=1:length(path.a0))
    validate_x_y_weights(X, y, weights)
    risk = exp(predict(path, X, model))
    devs = zeros(size(risk, 2))
	for j = 1:size(risk, 2)
		rsk = risk[:, j] .* weights;
		cumrsk = 0;
		for i = size(risk, 1):-1:1
			cumrsk += rsk[i]
			if y[i, 2] == 1
				dev[j] += log(rsk[i]) - log(cumrsk)
			end
		end
	end
    return -devs
end

loss(path::CoxNetPath, X::AbstractMatrix, y::Union(AbstractVector, AbstractMatrix),
     weights::AbstractVector=ones(size(y, 1)), va...) =loss(path, float64(X), float64(y),
                                                            float64(weights), va...)


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



function glmnet!(X::Matrix{Float64}, y::Matrix, family::CoxPL;
             offsets::Union(Vector{Float64}, Nothing)=nothing,
             weights::Vector{Float64}=ones(length(y)),
             alpha::Real=1.0,
             penalty_factor::Vector{Float64}=ones(size(X, 2)),
             constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
             dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), 
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
	ca = zeros(Float64, pmax, nlambda)
	ia = zeros(Int32, pmax)
	nin = zeros(Int32, nlambda)
	fdev = zeros(Float64, nlambda)
	alm = zeros(Float64, nlambda)
	#
    offsets = isa(offsets, Nothing) ? zeros(length(times)) : copy(offsets)
    length(offsets) == length(times) || error("length of offsets must match length of y")
	#
	ccall((:coxnet_, libglmnet), Void, 
		(
			Ptr{Float64} , Ptr{Int32}     , Ptr{Int32}        , Ptr{Float64} ,
			Ptr{Float64} , Ptr{Float64}   , Ptr{Float64}      , Ptr{Float64} ,
			Ptr{Int32}   , Ptr{Float64}   , Ptr{Float64}      , Ptr{Int32}   ,
			Ptr{Int32}   , Ptr{Int32}     , Ptr{Float64}      , Ptr{Float64} ,
			Ptr{Float64} , Ptr{Int32}     , Ptr{Int32}        , Ptr{Int32}   ,
			Ptr{Float64} , Ptr{Int32}     , Ptr{Int32}        , Ptr{Float64} ,
			Ptr{Float64} , Ptr{Float64}   , Ptr{Int32}        , Ptr{Int32}
		),
			&alpha       , &nobs          , &nvars            , X            ,
			times        , status         , offsets           , weights      ,
			&jd          , penalty_factor , constraints       , &dfmax       ,
			&pmax        , &nlambda       , &lambda_min_ratio , lambda       ,
			&tol         , &maxit         , &standardize      , lmu          ,
			ca           , ia             , nin               , &null_dev    ,
			fdev         , alm            , nlp               , jerr
		)
    @check_and_return_cox
end


glmnet(X::Matrix{Float64}, y::Matrix, family::CoxPL; kw...) =
    glmnet!(copy(X), copy(y), family; kw...)
