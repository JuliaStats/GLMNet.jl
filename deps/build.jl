cd(joinpath(Pkg.dir("GLMNet"), "deps"))
pic = @windows ? "" : "-fPIC"
run(`gfortran -m$WORD_SIZE -fdefault-real-8 -ffixed-form $pic -shared -O3 glmnet5.f90 -o libglmnet.so`)
