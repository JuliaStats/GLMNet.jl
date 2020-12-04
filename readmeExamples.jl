# all README.md results are from this script
using GLMNet, RDatasets, Plots, LinearAlgebra, LaTeXStrings
rng = MersenneTwister(1)
y = collect(1:100) + randn(rng,100)*10; X = [1:100 (1:100)+randn(rng,100)*5 (1:100)+randn(rng,100)*10 (1:100)+randn(rng,100)*20];
path = glmnet(X, y)
path.betas

# second block
using Plots, LinearAlgebra, LaTeXStrings
betaNorm = [norm(x, 1) for x in eachslice(path.betas,dims=2)];
extraOptions = (xlabel=L"\| \beta \|_1",ylabel=L"\beta_i", legend=:topleft,legendtitle="Variable", labels=[1 2 3 4]);
plot(betaNorm, path.betas'; extraOptions...)
savefig("regressionLassoPath.svg")
predict(path, [22 22+randn()*5 22+randn()*10 22+randn()*20])
cv = glmnetcv(X, y,rng =MersenneTwister(1))
argmin(cv.meanloss)
cv.path.betas[:, 59]
coef(cv)

# third block
using RDatasets
iris = dataset("datasets", "iris");
X = convert(Matrix, iris[:, 1:4]);
y = convert(Vector, iris[:Species]);
iTrain = sample(1:size(X,1), 100, replace = false);
iTest = setdiff(1:size(X,1), iTrain);
iris_cv = glmnetcv(X[iTrain, :], y[iTrain])
yht = round.(predict(iris_cv, X[iTest, :], outtype = :prob), digits=3);
DataFrame(target=y[iTest], set=yht[:,1], ver=yht[:,2], vir=yht[:,3])[5:5:50,:]
irisLabels = reshape(names(iris)[1:4],(1,4))
βs =iris_cv.path.betas
λs= iris_cv.lambda
sharedOpts =(legend=false,  xlabel=L"\lambda", xscale=:log10) 
p1 = plot(λs,βs[:,1,:]',ylabel=L"\beta_i";sharedOpts...)
p2 = plot(λs,βs[:,2,:]',title="Across Cross Validation runs";sharedOpts...)
p3 = plot(λs,βs[:,3,:]', legend=:topright,legendtitle="Variable", labels=irisLabels,xlabel=L"\lambda",xscale=:log10)
plot(p1,p2,p3,layout=(1,3))
savefig("iris_path.svg")

plot(iris_cv.lambda, iris_cv.meanloss, xscale=:log10, legend=false, yerror=iris_cv.stdloss,xlabel=L"\lambda",ylabel="loss")
vline!([lambdamin(iris_cv)])
savefig("lambda_plot.svg")
