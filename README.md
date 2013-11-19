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
[5,]      2  0.138376   0.0934088
[6,]      3  0.198957   0.0851106
[7,]      3  0.251511   0.0775496
[8,]      3  0.295142   0.0706603
[9,]      3  0.331366    0.064383
[10,]     3  0.361439   0.0586634
[11,]     3  0.386407   0.0534519
[12,]     3  0.407136   0.0487034
[13,]     3  0.424345   0.0443767
[14,]     3  0.438632   0.0404344
[15,]     3  0.450493   0.0368423
[16,]     3  0.460341   0.0335694
[17,]     3  0.468517   0.0305872
[18,]     3  0.475304   0.0278699
[19,]     3   0.48094    0.025394
[20,]     3  0.485618   0.0231381
  :
[45,]     5  0.520022  0.00226061
[46,]     5  0.520196  0.00205979
[47,]     5  0.520341   0.0018768
[48,]     5  0.520461  0.00171007
[49,]     5   0.52056  0.00155815
[50,]     5  0.520643  0.00141973
[51,]     5  0.520709  0.00129361
[52,]     5  0.520767  0.00117869
[53,]     5  0.520814  0.00107397
[54,]     5  0.520854 0.000978565
[55,]     5  0.520887 0.000891632
[56,]     5  0.520914 0.000812422
[57,]     5  0.520936 0.000740249
[58,]     5  0.520954 0.000674487
[59,]     5   0.52097 0.000614567
[60,]     5  0.520984 0.000559971
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
