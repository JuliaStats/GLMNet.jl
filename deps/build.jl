using Compat
using Libdl 

if Compat.Sys.iswindows()
    flags = ["-m$(Sys.WORD_SIZE)","-fdefault-real-8","-ffixed-form","-shared","-O3"]
else
    flags = ["-m$(Sys.WORD_SIZE)","-fdefault-real-8","-ffixed-form","-shared","-O3","-fPIC"]
end

run(`gfortran $flags glmnet5.f90 -o libglmnet.$(Libdl.dlext)`)
