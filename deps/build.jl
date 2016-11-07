using Compat
pic = @static is_windows() ? "" : "-fPIC"
run(`gfortran -m$(Sys.WORD_SIZE) -fdefault-real-8 -ffixed-form $pic -shared -O3 glmnet5.f90 -o libglmnet.$(Libdl.dlext)`)
