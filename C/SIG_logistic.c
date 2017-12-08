#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"
#include "mkl.h"
#define DEBUG 0
#define USE_BLAS 1

/*
SIG_logistic(w,Xt,y,lambda,eta,d,g);
% w(p,1) - updated in place
% wtilde(p,1) - updated in place
% G(p,1) - updated in place
% Xt(p,n) - real, can be sparse
% y(n,1) - {-1,1}
% lambda - scalar regularization param
% eta - scalar constant step size
% maxIter - maximal iterations of inner loop
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int nSamples, maxIter;
    int sparse = 0, useScaling = 1, useLazy=1,*lastVisited;
    long i, idx, j, nVars;

    mwIndex *jc, *ir;

    double *w, *wtilde, *G, *Xt, *y, lambda, eta, innerProdI, innerProdZ, tmpDelta, c = 1, tmpFactor, *cumSum;

    if (nrhs != 8)
        mexErrMsgTxt("Function needs 8 arguments");

    /* Input */

    w = mxGetPr(prhs[0]);
    wtilde = mxGetPr(prhs[1]);
    G = mxGetPr(prhs[2]);
    Xt = mxGetPr(prhs[3]);
    y = mxGetPr(prhs[4]);
    lambda = mxGetScalar(prhs[5]);
    eta = mxGetScalar(prhs[6]);
    maxIter = (int)mxGetScalar(prhs[7]);

    /* Compute Sizes */
    nVars = mxGetM(prhs[3]);
    nSamples = mxGetN(prhs[3]);


    if (nVars != mxGetM(prhs[0]))
        mexErrMsgTxt("w and Xt must have the same number of rows");
    if (nSamples != mxGetM(prhs[4]))
        mexErrMsgTxt("number of columns of Xt must be the same as the number of rows in y");

    srand(time(NULL));

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
        useLazy = 0;
    }

    if (sparse && eta * lambda == 1)
    {
        mexErrMsgTxt("Sorry, I don't like it when Xt is sparse and eta*lambda=1\n");
    }

    /* Allocate memory needed for lazy updates */
    if (useLazy)
    {
        lastVisited = mxCalloc(nVars,sizeof(int));
        cumSum = mxCalloc(maxIter,sizeof(double));
    }


#if DEBUG
    mexPrintf("maxIter: %d\n", maxIter);
#endif
    // @NOTE main loop
    for (i = 0; i < maxIter; i++)
    {
        // idx = rand() % nSamples;  // sample
        idx = i % nSamples;

        /* Step 1: Compute current values of needed parameters w_{i} */
        if (useLazy && i > 0)
        {
            for(j = jc[idx]; j < jc[idx+1]; j++)
            {
                if (lastVisited[ir[j]] == 0)
                {  // or we can let lastVisited[-1] = 0
                    w[ir[j]] -= G[ir[j]] * cumSum[i-1];
                }
                else { // if lastVisited[ir[j]] > 0
                    w[ir[j]] -= G[ir[j]] * (cumSum[i-1] - cumSum[lastVisited[ir[j]]-1]);
                }
                lastVisited[ir[j]] = i;
            }
        }

        /* Step 2:  Compute derivative of loss \nabla f(w_{i}) */
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

        // update cumSum
        if (useScaling)
        {
            c *= 1-eta*lambda;
            tmpFactor = eta / c;

            if (useLazy)
            {
                if (i == 0)
                    cumSum[0] = tmpFactor;
                else
                    cumSum[i] = cumSum[i-1] + tmpFactor;
            }
            else
            {
#if USE_BLAS
                cblas_daxpy(nVars, -tmpFactor, G, 1, w, 1);
#else
                for(j = 0; j < nVars; j++)
                {
                    w[j] -= tmpFactor * G[j];
                }
#endif
            }
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

        /* Step 3: approximate w_{i+1} */
        if (sparse)
        {
#if USE_BLAS_SPARSE
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

        // Re-normalize the parameter vector if it has gone numerically crazy
        if(c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) || (c < 0 && c > -1e-100))
        {

            if (useLazy)
            {
                for(j = 0; j < nVars; j++)
                {
                    if (lastVisited[j] == 0)
                        w[j] -= G[j] * cumSum[i];
                    else
                        w[j] -= G[j] * (cumSum[i]-cumSum[lastVisited[j]-1]);
                    lastVisited[j] = i+1;
                }
                cumSum[i] = 0;
            }
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
    }

    if (useLazy)
    {
        for(j = 0; j < nVars; j++)
        {
            if (lastVisited[j] == 0)
            {
                w[j] -= G[j] * cumSum[maxIter-1];
            }
            else
            {
                w[j] -= G[j] * (cumSum[maxIter-1] - cumSum[lastVisited[j]-1]);
            }
        }
    }

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

    if(useLazy)
    {
        mxFree(lastVisited);
        mxFree(cumSum);
    }
    return;
}
