#include <math.h>
#include <stdlib.h>
#include <time.h>
#include "mex.h"
#include "mkl.h"
#define DEBUG 0
#define DEBUGADDR 0
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
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int nSamples, maxIter;
    int sparse = 0, useScaling = 1;
    int i, idx, j, nVars;

    mwIndex *jc, *ir;

    double *w, *wtilde, *G, *Xt, *y, lambda, eta, innerProdI, innerProdZ, tmpDelta, c = 1, tmpFactor;

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

#if DEBUGADDR
    //printf("nVars: %d, nSamples: %d\n", nVars, nSamples);
    printf("w: %ld, wtilde: %ld\n", w, wtilde);
#endif

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
    }

    if (sparse && eta * lambda == 1)
        mexErrMsgTxt("Sorry, I don't like it when Xt is sparse and eta*lambda=1\n");
        // why not?

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
#if DEBUG
        printf("tmpDelta: %lf\n", tmpDelta);
        printf("w[0]: %lf\n", w[0]);
        printf("wtilde[0]: %lf\n", wtilde[0]);
#endif

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
            for(j = 0; j < nVars; j++)
            {
                w[j] *= 1 - eta * lambda;
                w[j] -= eta * G[j];
            }
            tmpFactor = eta * tmpDelta;
        }
#if DEBUG
        printf("w[0]: %lf\n", w[0]);
#endif

        if (sparse)
        {
            for(j = jc[idx]; j < jc[idx+1]; j++)
            {
                w[ir[j]] -= tmpFactor * Xt[j];
            }
        }
        else
        {
            for(j = 0; j < nVars; j++)
            {
                w[j] -= tmpFactor * Xt[j + nVars * idx];
            }
        }
#if DEBUG
        printf("w[0]: %lf\n", w[0]);
#endif

        /* Re-normalize the parameter vector if it has gone numerically crazy */
        if(c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) || (c < 0 && c > -1e-100))
        {
#if DEBUG
            printf("Oops, we have to re-nomalize...\n");
#endif
            for(j = 0; j < nVars; j++)
            {
                w[j] = c * w[j];
            }
            c = 1;
        }
    }

    if(useScaling)
    {
#if DEBUG
        printf("Oops, we are using scaling ...\n");
#endif
        for(j = 0; j < nVars; j++)
        {
            w[j] *= c;
        }
    }
    return;
}
