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
```

Classification example:

```julia
julia> using RDatasets

julia> iris = dataset("datasets", "iris");

julia> X = convert(Matrix, iris[:, 1:4]);

julia> y = convert(Vector, iris[:Species]);

julia> iris_cv = glmnetcv(X, y)
Multinomial GLMNet Cross Validation
100 models for 4 predictors in 10 folds
Best λ 0.000 (mean loss -13.129, std 8.670)

julia> xpred = predict(iris_cv, X, outtype = :prob)
150x3 Array{Float64,2}:
 0.33402      0.66598      8.18545e-9
 0.0547817    0.945218     4.31347e-7
 0.290808     0.709192     6.10286e-8
 0.116339     0.88366      1.07654e-6
 0.519054     0.480946     4.00958e-9
 0.170126     0.829874     1.84398e-7
 0.370725     0.629275     2.67971e-7
 0.197469     0.802531     5.72337e-8
 0.101571     0.898427     2.53714e-6
 ⋮
 1.32985e-36  4.43691e-22  1.0
 9.81275e-35  2.55675e-21  1.0
 1.13035e-40  5.2086e-25   1.0
 5.76539e-41  1.45827e-25  1.0
 1.1244e-37   6.02061e-23  1.0
 3.63512e-35  5.8802e-21   1.0
 9.03786e-35  6.55048e-21  1.0
 8.19343e-37  3.76175e-23  1.0
 2.91289e-32  1.19357e-19  1.0
```


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
