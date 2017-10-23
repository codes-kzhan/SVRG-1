#include <math.h>
#include <stdlib.h>
#include <time.h>
//#include "mex.h"
#include "/usr/local/MATLAB/R2017a/extern/include/mex.h"
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
% u(p,1) - updated in place
% z(p,1) - updated in place
% tau1 - parameter
% tau2 - parameter
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int nSamples, maxIter;
    int sparse = 0;
    long i, idx, j, nVars;

    mwIndex *jc, *ir;

    double *w, *wtilde, *G, *Xt, *y, lambda, eta, *u, *z, tau1, tau2, *znew, innerProdI, innerProdZ, tmpDelta, *tmpPtr;

    if (nrhs != 12)
        mexErrMsgTxt("Function needs 12 arguments");

    /* Input */

    w = mxGetPr(prhs[0]);
    wtilde = mxGetPr(prhs[1]);
    G = mxGetPr(prhs[2]);
    Xt = mxGetPr(prhs[3]);
    y = mxGetPr(prhs[4]);
    lambda = mxGetScalar(prhs[5]);
    eta = mxGetScalar(prhs[6]);
    maxIter = (int)mxGetScalar(prhs[7]);
    u = mxGetPr(prhs[8]);
    z = mxGetPr(prhs[9]);
    tau1 = mxGetScalar(prhs[10]);
    tau2 = mxGetScalar(prhs[11]);

    /* Compute Sizes */
    nVars = mxGetM(prhs[3]);
    nSamples = mxGetN(prhs[3]);


    if (nVars != mxGetM(prhs[0]))
        mexErrMsgTxt("w and Xt must have the same number of rows");
    if (nSamples != mxGetM(prhs[4]))
        mexErrMsgTxt("number of columns of Xt must be the same as the number of rows in y");
    if (nVars != mxGetM(prhs[8]))
        mexErrMsgTxt("w and u must have the same number of rows");
    if (nVars != mxGetM(prhs[9]))
        mexErrMsgTxt("w and z must have the same number of rows");

    srand(time(NULL));

    znew = mxCalloc(nVars, sizeof(double));
#if DEBUG
    mexPrintf("here we are\n");
#endif

    // sparse matrix uses scaling and lazy stuff
    if (mxIsSparse(prhs[3]))
    {
        sparse = 1;
        jc = mxGetJc(prhs[3]);
        ir = mxGetIr(prhs[3]);
    }

    if (sparse && eta * lambda == 1)
    {
        mexErrMsgTxt("Sorry, I don't like it when Xt is sparse and eta*lambda=1\n");
    }

    // @NOTE main loop
    for (i = 0; i < maxIter; i++)
    {
        idx = rand() % nSamples;  // sample
        //idx = i; // % nSamples;

        /* Step 1: update w */
#if USE_BLAS
        tmpPtr = w;
        w = u;
        u = tmpPtr;
        cblas_daxpby(nVars, tau2, wtilde, 1, 1 - tau1 - tau2, w, 1);
        cblas_daxpy(nVars, tau1, z, 1, w, 1);
#else
        // @TODO
#endif

        /* Step 2: calculate the new z */
        // compute derivative first
        innerProdI = 0;
        innerProdZ = 0;
        if (sparse)
        {
            for(j = jc[idx]; j < jc[idx+1]; j++)
            {
                innerProdI += w[ir[j]] * Xt[j];
                innerProdZ += wtilde[ir[j]] * Xt[j];
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
        tmpDelta = -eta * y[idx] / (1 + exp(y[idx] * innerProdI)) + y[idx] / (1 + exp(y[idx] * innerProdZ));
#if USE_BLAS
        cblas_dcopy(nVars, z, 1, znew, 1);
        cblas_daxpy(nVars, -eta, G, 1, znew, 1);
        cblas_daxpy(nVars, -eta*lambda, w, 1, znew, 1);

        //cblas_daxpyi(jc[idx+1] - jc[idx], -tmpFactor, Xt + jc[idx], (int *)(ir + jc[idx]), w);
        for(j = jc[idx]; j < jc[idx+1]; j++)
        {
            znew[ir[j]] -= tmpDelta * Xt[j];
        }
#else
        // @TODO
#endif

        /* Step 3: update u */
#if USE_BLAS
        tmpPtr = u;
        u = z;
        z = tmpPtr;
        cblas_daxpby(nVars, tau1, znew, 1, -tau1, u, 1);
        cblas_daxpy(nVars, 1, w, 1, u, 1);
#else
        // @TODO
#endif

        /* Step 4: update z */
        tmpPtr = z;
        z = znew;
        znew = tmpPtr;
    }
    return;
}
