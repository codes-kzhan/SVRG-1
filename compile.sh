mex -largeArrayDims C/SVRG_logistic.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/IAG_logistic.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

# mex -largeArrayDims C/SAG_logistic.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/DIG_logistic.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl
