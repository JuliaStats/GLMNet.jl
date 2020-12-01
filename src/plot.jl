import Gadfly.plot, Gadfly.Geom, Gadfly.Scale, Gadfly.Guide, Gadfly.mm
export plot

function plot(path::Union(GLMNetPath, CoxNetPath), args...; x = :lambda, y = :coefficients, color = :variable, xgroup = :response, kw...)
    betas = convert(Array, path.betas)
    nlambda = size(betas, 2)
    nvar = size(betas, 1)
    df = DataFrame()
    df[:variable] = rep(map(string, [1:nvar]), nlambda)

    if y == :coefficients || y == :coef
		y = :coefficients
        df[:coefficients] = betas[:]
    elseif y == :absCoefficients || y == :absCoef
		y = :absCoefficients
        df[:absCoefficients] = abs(betas[:])
    else
        error("y must be either :coef/:coefficients or :absCoef/:absCoefficients.")
    end

    if x == :lambda
        df[:lambda] = rep(path.lambda, each = nvar)
    elseif x == :deviance || x == :dev
		x = :deviance
        df[:deviance] = rep(path.dev_ratio, each = nvar)
    elseif x == :norm1
        df[:norm1] = rep(sum(abs(betas), 1)[:], each = nvar)
    elseif x == :norm2
        df[:norm2] = rep(sqrt(sum(abs(betas).^2, 1))[:], each = nvar)
    else
        error("x must be either :lambda, :dev/:deviance, :norm1, :norm2.")
    end
    plot(df, Geom.line, args...; x = x, y = y, color = color, kw...)
end

function plot(path::LogNetPath, args...; x = :lambda, y = :coefficients, color = :variable, xgroup = :response, kw...)
    nlambda = size(path.betas, 3)
    nresp = size(path.betas, 2)
    nvar = size(path.betas, 1)
    df = DataFrame()
    df[:response] = rep(rep(map(string, [1:nresp]), each = nvar), nlambda)
    df[:variable] = rep(map(string, [1:nvar]), nlambda*nresp)

    if y == :coefficients || y == :coef
		y = :coefficients
        df[:coefficients] = path.betas[:]
    elseif y == :absCoefficients || y == :absCoef
		y = :absCoefficients
        df[:absCoefficients] = abs(path.betas[:])
    else
        error("y must be either :coef/:coefficients or :absCoef/:absCoefficients.")
    end

    if x == :lambda
        df[:lambda] = rep(path.lambda, each = nvar*nresp)
    elseif x == :deviance || x == :dev
		x = :deviance
        df[:deviance] = rep(path.dev_ratio, each = nvar*nresp)
    elseif x == :norm1
        df[:norm1] = rep(sum(abs(path.betas), 1)[:], each = nvar)
    elseif x == :norm2
        df[:norm2] = rep(sqrt(sum(abs(path.betas).^2, 1))[:], each = nvar)
    else
        error("x must be either :lambda, :dev/:deviance, :norm1, :norm2.")
    end

    plot(df, Geom.subplot_grid(Geom.line, free_x_axis = true), args...;  x = x, y = y, color = color, xgroup = xgroup, kw...)
end


function plot(pathcv::GLMNetCrossValidation, args...)
    df = DataFrame(lambda = pathcv.lambda, deviance = pathcv.meanloss,
               lower = pathcv.meanloss - pathcv.stdloss, 
               upper = pathcv.meanloss + pathcv.stdloss);
    plot(df, Scale.x_log10, Geom.line, Geom.errorbar, Geom.vline(color = "orange", size = 0.5mm),
        x = :lambda, y = :deviance, ymin = :lower, ymax = :upper, xintercept = [lambdamin(pathcv)], args...)
end
