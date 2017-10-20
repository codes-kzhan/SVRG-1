#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"
#include "mkl.h"
#define DEBUG 0
#define USE_BLAS 1

/*
SVRG_logistic(w,Xt,y,lambda,eta,d,g);
% w(p,1) - updated in place
% wtilde(p,1) - updated in place
% G(p,1) - updated in place
% Xt(p,n) - real, can be sparse
% y(n,1) - {-1,1}
% lambda - scalar regularization param
% eta - scalar constant step size
% maxIter - maximal iterations of inner loop
% passes - maximal epochs
% subOptimality - store each epoch's error
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int nSamples, maxIter, passes;
    int sparse = 0, useScaling = 1;
    long i, idx, j, nVars, k;

    mwIndex *jc, *ir;

    double *w, *wtilde, *G, *Xt, *y, lambda, eta, *subOptimality, innerProdI, innerProdZ, tmpDelta, c = 1, tmpFactor;

    if (nrhs != 10)
        mexErrMsgTxt("Function needs 10 arguments");

    /* Input */

    w = mxGetPr(prhs[0]);
    wtilde = mxGetPr(prhs[1]);
    G = mxGetPr(prhs[2]);
    Xt = mxGetPr(prhs[3]);
    y = mxGetPr(prhs[4]);
    lambda = mxGetScalar(prhs[5]);
    eta = mxGetScalar(prhs[6]);
    maxIter = (int)mxGetScalar(prhs[7]);
    passes = (int)mxGetScalar(prhs[8]);
    subOptimality = mxGetPr(prhs[9]);

    /* Compute Sizes */
    nVars = mxGetM(prhs[3]);
    nSamples = mxGetN(prhs[3]);


    if (nVars != mxGetM(prhs[0]))
        mexErrMsgTxt("w and Xt must have the same number of rows");
    if (nSamples != mxGetM(prhs[4]))
        mexErrMsgTxt("number of columns of Xt must be the same as the number of rows in y");

    srand(time(NULL));
    //printf("size of index: %d, size of int: %d\n", sizeof(mwIndex), sizeof(int));

    // sparse matrix uses scaling and lazy stuff
    if (mxIsSparse(prhs[3]))
    {
        sparse = 1;
        jc = mxGetJc(prhs[3]);
        ir = mxGetIr(prhs[3]);
    }
    else
    {
        useScaling = 0;
    }

    if (sparse && eta * lambda == 1)
        mexErrMsgTxt("Sorry, I don't like it when Xt is sparse and eta*lambda=1\n");
        // why not?

    for (k = 1; k <= passes; k++)  // start from the first pass
    {
        // compute full gradient G
        // @TODO

        for (i = 0; i < maxIter; i++)
        {
            //i = k % nSamples; // deterministic order
            idx = rand() % nSamples;  // sample

            /* Compute derivative of loss */
            innerProdI = 0;
            innerProdZ = 0;
            if (sparse)
            {
                for(j = jc[idx]; j < jc[idx+1]; j++)
                {
                    innerProdI += w[ir[j]] * Xt[j];
                    innerProdZ += wtilde[ir[j]] * Xt[j];
                }
                if (useScaling)
                {
                    innerProdI *= c;
                }
            }
            else
            {
                for(j = 0; j < nVars; j++)
                {
                    innerProdI += w[j] * Xt[j + nVars * idx];
                    innerProdZ += wtilde[j] * Xt[j + nVars * idx];
                }
            }
            tmpDelta = -y[idx] / (1 + exp(y[idx] * innerProdI)) + y[idx] / (1 + exp(y[idx] * innerProdZ));

            /* Update parameters */
            if (useScaling)
            {
                c *= 1-eta*lambda;
                tmpFactor = eta / c;

#if USE_BLAS
                cblas_daxpy(nVars, -tmpFactor, G, 1, w, 1);
#else
                for(j = 0; j < nVars; j++)
                {
                    w[j] -= tmpFactor * G[j];
                }
#endif
                tmpFactor = eta / c * tmpDelta; // tmpFactor is used for next if-else
            }
            else
            {
#if USE_BLAS
                cblas_daxpby(nVars, -eta, G, 1, 1 - eta * lambda, w, 1);
#else
                tmpFactor = 1 - eta * lambda;
                for(j = 0; j < nVars; j++)
                {
                    w[j] *= tmpFactor;
                    w[j] -= eta * G[j];
                }
#endif
                tmpFactor = eta * tmpDelta;
            }

            if (sparse)
            {
#if USE_BLAS
                cblas_daxpyi(jc[idx+1] - jc[idx], -tmpFactor, Xt + jc[idx], (int *)(ir + jc[idx]), w);
                // @NOTE (int *) here is 64bit because mwIndex is 64bit, and we have to link libmkl_intel_ilp64.a for 64bit integer
#else
                for(j = jc[idx]; j < jc[idx+1]; j++)
                {
                    w[ir[j]] -= tmpFactor * Xt[j];
                }
#endif
            }
            else
            {
#if USE_BLAS
                cblas_daxpy(nVars, -tmpFactor, Xt + nVars * idx, 1, w, 1);
#else
                for(j = 0; j < nVars; j++)
                {
                    w[j] -= tmpFactor * Xt[j + nVars * idx];
                }
#endif
            }

            /* Re-normalize the parameter vector if it has gone numerically crazy */
            if(c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) || (c < 0 && c > -1e-100))
            {

#if USE_BLAS
                cblas_dscal(nVars, c, w, 1);
#else
                for(j = 0; j < nVars; j++)
                {
                    w[j] = c * w[j];
                }
#endif
                c = 1;
            }
        }  // iteratons

        if(useScaling)
        {
#if USE_BLAS
            cblas_dscal(nVars, c, w, 1);
#else
            for(j = 0; j < nVars; j++)
            {
                w[j] *= c;
            }
#endif
        }

            // copy w into wtilde
#if USE_BLAS
            cblas_dcopy(nVars, w, 1, wtilde, 1);
#else
            for(j = 0; j < nVars; j++)
            {
                wtilde[j] = w[j];
            }
#endif
            // compute cost and distance to the optimal
            // @TODO

    }  // passes
    return;
}
