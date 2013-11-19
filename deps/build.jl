using BinDeps
cd(joinpath(Pkg.dir("Glmnet"), "deps"))
run(`gfortran -fdefault-real-8 -ffixed-form -fPIC -shared -O3 glmnet3.f90 -o libglmnet.so`)
