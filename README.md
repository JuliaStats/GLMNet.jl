# GLMNet

[![Build Status](https://travis-ci.org/simonster/GLMNet.jl.svg?branch=master)](https://travis-ci.org/simonster/GLMNet.jl)
[![Coverage Status](https://coveralls.io/repos/simonster/GLMNet.jl/badge.svg?branch=master)](https://coveralls.io/r/simonster/GLMNet.jl?branch=master)

[glmnet](http://www.jstatsoft.org/v33/i01/) is an R package by Jerome Friedman, Trevor Hastie, Rob Tibshirani that fits entire Lasso or ElasticNet regularization paths for linear, logistic, multinomial, and Cox models using cyclic coordinate descent. This Julia package wraps the Fortran code from glmnet.

## Quick start

To fit a basic regression model:

```julia
julia> using GLMNet

julia> srand(123)

julia> y = [1:100]+randn(100)*10;

julia> X = [1:100 (1:100)+randn(100)*5 (1:100)+randn(100)*10 (1:100)+randn(100)*20];

julia> path = glmnet(X, y)
Least Squares GLMNet Solution Path (74 solutions for 4 predictors in 832 passes):
74x3 DataFrame
| Row | df | pct_dev  | λ         |
|-----|----|----------|-----------|
| 1   | 0  | 0.0      | 29.6202   |
| 2   | 1  | 0.148535 | 26.9888   |
| 3   | 1  | 0.271851 | 24.5912   |
| 4   | 1  | 0.37423  | 22.4066   |
⋮
| 70  | 4  | 0.882033 | 0.0482735 |
| 71  | 4  | 0.882046 | 0.043985  |
| 72  | 4  | 0.882058 | 0.0400775 |
| 73  | 4  | 0.882067 | 0.0365171 |
| 74  | 4  | 0.882075 | 0.033273  |
```

`path` represents the Lasso or ElasticNet fits for varying values of λ. The value of the intercept for each λ value are in `path.a0`. The coefficients for each fit are stored in compressed form in `path.betas`.

```julia
julia> path.betas
4x74 CompressedPredictorMatrix:
 0.0  0.091158  0.174218  0.249899  0.318857  0.381688  0.438938  0.491102  0.538632  …   0.902207   0.905364   0.908404   0.910988   0.913497   0.915593   0.917647
 0.0  0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0           0.129457   0.129059   0.128679   0.128359   0.128054   0.127805   0.127568
 0.0  0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0          -0.116622  -0.119305  -0.121874  -0.12408   -0.126211  -0.128015  -0.129776
 0.0  0.0       0.0       0.0       0.0       0.0       0.0       0.0       0.0           0.108096   0.108137   0.108159   0.108198   0.108217   0.108254   0.108272
```

This CompressedPredictorMatrix can be indexed as any other AbstractMatrix, or converted to a Matrix using `convert(Matrix, path.betas)`.

One can visualize the path by

```julia
julia> using Gadfly

julia> plot(path, Scale.x_log10, Guide.xlabel("λ"))
```
![regression-lasso-path](https://raw.githubusercontent.com/linxihui/Misc/master/Images/GLMNet.jl/regression_lasso_path.png)

To predict the output for each model along the path for a given set of predictors, use `predict`:

```julia
julia> predict(path, [22 22+randn()*5 22+randn()*10 22+randn()*20])
1x74 Array{Float64,2}:
 50.8669  48.2689  45.9017  43.7448  41.7795  39.9888  38.3572  36.8705  35.5159  …  21.9056  21.9115  21.9171  21.922  21.9266  21.9306  21.9344  21.9377  21.9407
```

To find the best value of λ by cross-validation, use `glmnetcv`:

```julia
julia> cv = glmnetcv(X, y)
Least Squares GLMNet Cross Validation
74 models for 4 predictors in 10 folds
Best λ 0.450 (mean loss 129.720, std 14.871)

julia> indmin(cv.meanloss)
46

julia> cv.path.betas[:, 46]
4-element Array{Float64,1}:
 0.781119
 0.128094
 0.0
 0.103008

julia> coef(cv)
4-element Array{Float64,1}:
 0.781119
 0.128094
 0.0
 0.103008

julia> plot(cv)
```
![regression-cv](https://raw.githubusercontent.com/linxihui/Misc/master/Images/GLMNet.jl/regression_cv.png)

Classification example:

```julia
julia> using RDatasets

julia> iris = dataset("datasets", "iris");

julia> X = convert(Matrix, iris[:, 1:4]);

julia> y = convert(Vector, iris[:Species]);

julia> iTrain = sample(1:size(X,1), 100, replace = false);

julia> iTest = setdiff(1:size(X,1), iTrain);

julia> iris_cv = glmnetcv(X[iTrain, :], y[iTrain])
Multinomial GLMNet Cross Validation
100 models for 4 predictors in 10 folds
Best λ 0.000 (mean loss -2.195, std 0.384)

julia> xpred = predict(iris_cv, X[iTest, :], outtype = :prob);
julia> convert(DataFrame, [y[iTest] round(xpred, 3)])
50x4 DataFrame
| Row | x1          | x2    | x3    | x4  |
|-----|-------------|-------|-------|-----|
| 1   | "setosa"    | 0.275 | 0.725 | 0.0 |
| 2   | "setosa"    | 0.763 | 0.237 | 0.0 |
| 3   | "setosa"    | 0.414 | 0.586 | 0.0 |
| 4   | "setosa"    | 0.146 | 0.854 | 0.0 |
| 5   | "setosa"    | 0.568 | 0.432 | 0.0 |
| 6   | "setosa"    | 0.76  | 0.24  | 0.0 |
| 7   | "setosa"    | 0.864 | 0.136 | 0.0 |
| 8   | "setosa"    | 0.831 | 0.169 | 0.0 |
| 9   | "setosa"    | 0.663 | 0.337 | 0.0 |
| 10  | "setosa"    | 0.035 | 0.965 | 0.0 |
⋮
| 40  | "virginica" | 0.0   | 0.0   | 1.0 |
| 41  | "virginica" | 0.0   | 0.0   | 1.0 |
| 42  | "virginica" | 0.0   | 0.0   | 1.0 |
| 43  | "virginica" | 0.0   | 0.0   | 1.0 |
| 44  | "virginica" | 0.0   | 0.0   | 1.0 |
| 45  | "virginica" | 0.0   | 0.0   | 1.0 |
| 46  | "virginica" | 0.0   | 0.0   | 1.0 |
| 47  | "virginica" | 0.0   | 0.0   | 1.0 |
| 48  | "virginica" | 0.0   | 0.0   | 1.0 |
| 49  | "virginica" | 0.0   | 0.0   | 1.0 |
| 50  | "virginica" | 0.0   | 0.0   | 1.0 |

julia> plot(iris_cv.path)
```
![iris-lasso-path](https://raw.githubusercontent.com/linxihui/Misc/master/Images/GLMNet.jl/iris_lasso_path.png) 


## Fitting models

`glmnet` has two required parameters: the m x n predictor matrix `X` and the dependent variable `y`. It additionally accepts an optional third argument, `family`, which can be used to specify a generalized linear model. Currently, `Normal()` (least squares, default), `Binomial()` (logistic), `Poisson()` , `Multinomial()`, `CoxPH()` (Cox model) are supported. 

- For linear and Poisson models, `y` is a numerical vector.
- For logistic models, `y` is either a string vector or a m x 2 matrix, where the first column is the count of negative responses for each row in `X` and the second column is the count of positive responses. 
- For multinomial models, `y` is etiher a string vector (with at least 3 unique values) or a m x k matrix, where k is number of unique values (classes).
- For Cox models, `y` is a 2-column matrix, where the first column is survival time and second column is (right) censoring status. Indeed, For survival data, `glmnet` has another method `glmnet(X::Matrix, time::Vector, status::Vector)`. Same for `glmnetcv`.


`glmnet` also accepts many optional keyword parameters, described below:

 - `weights`: A vector of weights for each sample of the same size as `y`.
 - `alpha`: The tradeoff between lasso and ridge regression. This defaults to `1.0`, which specifies a lasso model.
 - `penalty_factor`: A vector of length n of penalties for each predictor in `X`. This defaults to all ones, which weights each predictor equally. To specify that a predictor should be unpenalized, set the corresponding entry to zero.
 - `constraints`: An n x 2 matrix specifying lower bounds (first column) and upper bounds (second column) on each predictor. By default, this is `[-Inf Inf]` for each predictor in `X`.
 - `dfmax`: The maximum number of predictors in the largest model.
 - `pmax`: The maximum number of predictors in any model.
 - `nlambda`: The number of values of λ along the path to consider.
 - `lambda_min_ratio`: The smallest λ value to consider, as a ratio of the value of λ that gives the null model (i.e., the model with only an intercept). If the number of observations exceeds the number of variables, this defaults to `0.0001`, otherwise `0.01`.
 - `lambda`: The λ values to consider. By default, this is determined from `nlambda` and `lambda_min_ratio`.
 - `tol`: Convergence criterion. Defaults to `1e-7`.
 - `standardize`: Whether to standardize predictors so that they are in the same units. Defaults to `true`. Beta values are always presented on the original scale.
 - `intercept`: Whether to fit an intercept term. The intercept is always unpenalized. Defaults to `true`.
 - `maxit`: The maximum number of iterations of the cyclic coordinate descent algorithm. If convergence is not achieved, a warning is returned.


## See also

 - [Lasso.jl](https://github.com/simonster/Lasso.jl), a pure Julia implementation of the glmnet coordinate descent algorithm that often achieves better performance.
 - [LARS.jl](https://github.com/simonster/LARS.jl), an implementation
   of least angle regression for fitting entire linear (but not
   generalized linear) Lasso and Elastic Net coordinate paths.
