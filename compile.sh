mex -largeArrayDims C/SVRG_logistic.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/IAG_logistic.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

# mex -largeArrayDims C/SAG_logistic.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/DIG_logistic.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

# mex -largeArrayDims C/Katyusha_logistic.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/Katyusha_sparse.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/SIG_logistic.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/SIGM_sparse.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/SVRG_svm.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/Katyusha_svm.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/SIG_svm.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/SIGM_svm.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/DIG_svm.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl

mex -largeArrayDims C/IAG_svm.c ${MKLROOT}/lib/intel64/libmkl_intel_ilp64.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a ${MKLROOT}/lib/intel64/libmkl_core.a -liomp5 -lpthread -lm -ldl
