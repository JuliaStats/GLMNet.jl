import Distributions.MvNormal

MvNormal() = MvNormal([0, 0], [1 0; 0 1]) # MvNormal(0, ([1;]))
modeltype(::MvNormal) = "MvNormal"


function predict(path::GLMNetPath{<:MvNormal}, X::AbstractMatrix,
    model::Union{Int, AbstractVector{Int}}=1:length(path.lambda); 
    outtype = :link, offsets = zeros(size(X, 1), size(path.betas, 2)))
    
    nresp = size(path.betas, 2);
    out = zeros(Float64, size(X, 1), nresp, length(model));
    for i = 1:length(model)
       out[:, :, i] = repeat(path.a0[:,model[i]]', size(X, 1)) + X * path.betas[:, :, model[i]] + offsets
    end
    if outtype != :link
       for i = 1:size(X, 1), j = 1:length(model)
           out = exp.(out)
       end
    end
    if length(model) == 1
       return out[:, :, 1]
    else
       return out
    end
end

nactive(g::GLMNetPath{<:MvNormal}, b::AbstractVector{Int}=1:size(g.betas, 3)) =
    [nactive(g.betas, j, dims=2) for j in b]

function show(io::IO, g::GLMNetPath{<:MvNormal})
    println(io, "$(modeltype(g.family)) GLMNet Solution Path ($(size(g.betas, 3)) solutions for $(size(g.betas, 1)) predictors in $(g.npasses) passes):")
    print(io, DataFrame(df=nactive(g), pct_dev=g.dev_ratio, λ=g.lambda))
end

struct MultiMSE <: Loss
    y::Matrix{Float64}
end
loss(l::MultiMSE, i, mu) = sum(abs2.(l.y[i,:] .- mu))

devloss(::MvNormal, y) = MultiMSE(y)

function loss(path::GLMNetPath{<:MvNormal}, X::AbstractMatrix{Float64},
    y::Union{AbstractVector{Float64}, AbstractMatrix{Float64}},
    weights::AbstractVector{Float64}=ones(size(y,1)),
    lossfun::Loss=devloss(path.family, y),
    model::Union{Int, AbstractVector{Int}}=1:length(path.lambda);
    offsets = zeros(size(X, 1), size(path.betas, 2)))

    validate_x_y_weights(X, y, weights)
    mu = predict(path, X; offsets = offsets)
    devs = zeros(size(mu, 3))
    for j = 1:size(mu, 3), i = 1:size(mu, 1)
        devs[j] += loss(lossfun, i, vec(mu[i, :, j]))*weights[i]
    end
    devs/sum(weights)
end

loss(path::GLMNetPath{<:MvNormal}, X::AbstractMatrix, y::Union{AbstractVector, AbstractMatrix}, 
        weights::AbstractVector=ones(size(y,1)), va...; kw...) =
    loss(path, convert(Matrix{Float64}, X), 
        convert(Array{Float64}, y),
        convert(Vector{Float64}, weights), va...; kw...)



macro check_and_return_mvnormal()
    esc(quote
        check_jerr(jerr[], maxit,pmax)
        lmu = lmu_ref[]
        # first lambda is infinity; changed to entry point
        if isempty(lambda) && length(alm) > 2
            alm[1] = exp(2*log(alm[2])-log(alm[3]))
        end
        GLMNetPath(family, a0[:, 1:lmu], ca[sortperm(ia), :, 1:lmu],
                    null_dev[], fdev[1:lmu], alm[1:lmu], Int(nlp[]))
    end)
end

# change of parameters from elnet to multelnet
# ka,parm,no,ni,nr, x,y,w,jd, vp,cl,ne,nx, nlam,flmin,ulam,thr, isd,   ,intr,maxit,lmu, a0,ca,ia,nin, rsq,alm,nlp,jerr
#    parm,no,ni,nr, x,y,w,jd, vp,cl,ne,nx, nlam,flmin,ulam,thr, isd,jsd,intr,maxit,lmu, a0,ca,ia,nin, rsq,alm,nlp,jerr
#  multi-response normal 
function glmnet!(X::Matrix{Float64}, y::Matrix{Float64},
    family::MvNormal=MvNormal();
    weights::Vector{Float64}=ones(size(y,1)),
    naivealgorithm::Bool=(size(X, 2) >= 500), alpha::Real=1.0,
    penalty_factor::Vector{Float64}=ones(size(X, 2)),
    constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
    dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
    lambda_min_ratio::Real=(size(y, 1) < size(X, 2) ? 1e-2 : 1e-4),
    lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
    standardize_response::Bool=false,
    intercept::Bool=true, maxit::Int=1000000)
    
    @validate_and_init_multi
    standardize_response = Int32(standardize_response)

    # Compute null deviance
    yw = y .* repeat(weights, 1, size(y, 2))
    mu = mean(y, dims=1)
    if intercept == 0
        mu = fill(intercept, 1, size(y,2))
    end
    # Sum of squared error (weighted by obervation weights)
    null_dev[] = sum(weights) .* mean(abs2.(yw .- mu))


    ccall((:multelnet_, libglmnet), Cvoid,
        
        (Ref{Float64}, Ref{Int32}, Ref{Int32}, Ref{Int32}, 
        Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, 
        Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
        Ref{Int32}, Ref{Float64}, Ref{Float64}, Ref{Float64},
        Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32},
        Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
        Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}),
        
        alpha, nobs, nvars, nresp,
        X, y, weights, jd,
        penalty_factor, constraints, dfmax, pmax,
        nlambda, lambda_min_ratio, lambda, tol,
        standardize, standardize_response, intercept, maxit, lmu_ref,
        a0, ca, ia, nin,
        fdev, alm, nlp, jerr
    )

    @check_and_return_mvnormal
end



# multi-response sparse normal
function glmnet!(X::AbstractSparseMatrix{Float64}, y::Matrix{Float64},
    family::MvNormal=MvNormal();
    weights::Vector{Float64}=ones(size(y, 1)),
    naivealgorithm::Bool=(size(X, 2) >= 500), alpha::Real=1.0,
    penalty_factor::Vector{Float64}=ones(size(X, 2)),
    constraints::Array{Float64, 2}=[x for x in (-Inf, Inf), y in 1:size(X, 2)],
    dfmax::Int=size(X, 2), pmax::Int=min(dfmax*2+20, size(X, 2)), nlambda::Int=100,
    lambda_min_ratio::Real=(size(y, 1) < size(X, 2) ? 1e-2 : 1e-4),
    lambda::Vector{Float64}=Float64[], tol::Real=1e-7, standardize::Bool=true,
    standardize_response::Bool=false,
    intercept::Bool=true, maxit::Int=1000000)
    
    @validate_and_init_multi
    standardize_response = Int32(standardize_response)

    # Compute null deviance
    yw = y .* repeat(weights, 1, size(y, 2))
    mu = mean(y, dims=1)
    if intercept == 0
        mu = fill(intercept, 1, size(y,2))
    end
    # Sum of squared error (weighted by obervation weights)
    null_dev[] = sum(weights) .* mean(abs2.(yw .- mu))


    ccall((:multspelnet_, libglmnet), Cvoid,
        
        (Ref{Float64}, Ref{Int32}, Ref{Int32}, Ref{Int32}, 
        Ref{Float64}, Ref{Int32}, Ref{Int32}, Ref{Float64}, Ref{Float64}, Ref{Int32}, 
        Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
        Ref{Int32}, Ref{Float64}, Ref{Float64}, Ref{Float64},
        Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32},
        Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
        Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}),
        
        alpha, nobs, nvars, nresp,
        X.nzval, X.colptr, X.rowval, y, weights, jd,
        penalty_factor, constraints, dfmax, pmax,
        nlambda, lambda_min_ratio, lambda, tol,
        standardize, standardize_response, intercept, maxit, lmu_ref,
        a0, ca, ia, nin,
        fdev, alm, nlp, jerr
    )

    @check_and_return_mvnormal
end

glmnet(X::Matrix{Float64}, y::Matrix{Float64}, family::MvNormal; kw...) =
    glmnet!(copy(X), copy(y), family; kw...)
glmnet(X::SparseMatrixCSC{Float64,Int32}, y::Matrix{Float64}, family::MvNormal; kw...) =
    glmnet!(copy(X), copy(y), family; kw...)
glmnet(X::AbstractMatrix, y::AbstractMatrix{<:Number}, family::MvNormal; kw...) =
    glmnet(convert(Matrix{Float64}, X), convert(Matrix{Float64}, y), family; kw...)
glmnet(X::SparseMatrixCSC, y::AbstractMatrix{<:Number}, family::MvNormal; kw...) =
    glmnet(convert(SparseMatrixCSC{Float64,Int32}, X), convert(Matrix{Float64}, y), family; kw...)