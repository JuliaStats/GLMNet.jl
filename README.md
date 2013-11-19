# Glmnet

[![Build Status](https://travis-ci.org/simonster/Glmnet.jl.png)](https://travis-ci.org/simonster/Glmnet.jl)

[glmnet](http://www.jstatsoft.org/v33/i01/) is an R package by Jerome Friedman, Trevor Hastie, Rob Tibshirani that fits entire Lasso or ElasticNet regularization paths for linear, logistic, multinomial, and Cox models using cyclic coordinate descent. This Julia package wraps the Fortran code from glmnet.

## Quick start

To fit a basic model:

```julia
julia> fit(rand(10, 5), rand(10))
Glmnet Solution Path (64 solutions for 5 predictors in 333 passes):
64x3 DataFrame:
         df      %dev           λ
[1,]      0       0.0     0.13552
[2,]      1 0.0436258    0.123481
[3,]      1 0.0798447    0.112511
[4,]      1  0.109914    0.102516
  :
[61,]     5  0.520995 0.000510225
[62,]     5  0.521004 0.000464898
[63,]     5  0.521011 0.000423597
[64,]     5  0.521015 0.000385966


julia> ans.betas
5x59 CompressedPredictorMatrix:
 0.0   0.0          0.0        …  -0.0204835  -0.0205189  -0.0205512
 0.0   0.0          0.0           -0.0581614  -0.0582582  -0.0583464
 0.0   0.0101051    0.0210017      0.131757    0.131806    0.131851 
 0.0   0.0          0.0           -0.0668845  -0.0669516  -0.0670127
 0.0  -0.00992844  -0.0184835     -0.110947   -0.111015   -0.111077
```

The intercepts along the solution path can be accessed as `path.a0`.

## TODO

#### Soon
- Document all of glmnet's options here
- Logistic, multinomial, and Cox models
- Cross-validation

#### Later
- Sparse predictor matrices
- Multiple responses

#### Someday
- Non-canonical link functions
