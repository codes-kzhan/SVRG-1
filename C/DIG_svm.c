#include <math.h>
#include "mex.h"
#include "comm.h"
#define DEBUG 0

/*
 * SGD_logistic(w,Z,lambda,stepSizes,iVals,maxNorm,average);
 w(p,1) - updated in place
 Z(p,n) - real, can be sparse
 lambda - scalar regularization param
 passes - used to determine step sizes
 maxIter - iterations
 factor - used to determine step sizes
 */


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int k , nSamples, maxIter, passes, sparse=0, useScaling=1;
    long i, j, nVars;

    mwIndex *jc,*ir;

    double *w, *Z, lambda, innerProd, alpha, factor, sig, c=1;

    /* Input */

    if (nrhs < 6)
        mexErrMsgTxt("At least 6 arguments are needed: {w, Z, lambda, passes, maxIter, factor}");

    w = mxGetPr(prhs[0]);
    Z = mxGetPr(prhs[1]);
    lambda = mxGetScalar(prhs[2]);
    passes = (int)mxGetScalar(prhs[3]);
    maxIter = (int)mxGetScalar(prhs[4]);
    factor = mxGetScalar(prhs[5]);

    /* Compute Sizes */
    nVars = mxGetM(prhs[1]);
    nSamples = mxGetN(prhs[1]);

    if (nVars != mxGetM(prhs[0]))
        mexErrMsgTxt("w and Z must have the same number of rows");

    if (mxIsSparse(prhs[1])) {
        sparse = 1;
        jc = mxGetJc(prhs[1]);
        ir = mxGetIr(prhs[1]);
    }

#if DEBUG
    printf("maxIter: %d\n", maxIter);
    printf("sparse : %d\n", sparse);
#endif

    // main loop
    for(k = 0; k < maxIter; k++)
    {
        /* Select next training example */
        i = k % nSamples;  // counting from zero

        /* Compute Inner Product of Parameters with Features */
        innerProd = 0;
        if(sparse)
        { // @NOTE compute inner product for sparse matrix
            for(j = jc[i]; j < jc[i+1]; j++)
            {
                innerProd += w[ir[j]]*Z[j];
            }
            if (useScaling)
            {
                innerProd *= c;  // this is the real inner product
            }
        }
        else
        { // @NOTE compute inner product for dense matrix
            for(j = 0; j < nVars; j++)
            {
                innerProd += w[j] * Z[j + nVars*i];
            }
        }
        sig = 2 * max(1 + innerProd, 0);

        alpha = factor /((passes - 1) * maxIter + k + 1); // /lambda;
        //alpha = factor / (passes + 1);
#if DEBUG
        printf("alpha: %lf\n", alpha);
        //break;
#endif

        /* Update parameters */
        if (sparse)
        {
            if (useScaling)
            {
                if (alpha*lambda != 1)
                {
                    c *= 1-alpha*lambda;
                }
                else
                {
                    c = 1;
                    for(j = 0; j < nVars; j++)
                    {
                        w[j] = 0;
                    }
                }
                for(j = jc[i]; j < jc[i+1]; j++)
                {
                    w[ir[j]] -= alpha*Z[j]*sig/c;
                }
            }
            else
            {
                for(j = 0; j < nVars; j++)
                {
                    w[j] *= (1-alpha*lambda);
                }
                for(j = jc[i]; j < jc[i+1]; j++)
                {
                    w[ir[j]] -= alpha*Z[j]*sig;
                }
            }
        }
        else
        {
            for(j = 0; j < nVars; j++)
            {
                w[j] *= 1-alpha*lambda;
            }
            for(j = 0; j < nVars; j++)
            {
                w[j] -= alpha*Z[j + i*nVars]*sig;
            }
        }

#if DEBUG
        printf("c: %lf\n", c);
        printf("maxIter: %d\n", maxIter);
        printf("k: %d\n", k);
        //break;
#endif

        /* Re-normalize the parameter vector if it has gone numerically crazy */
        if(c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) || (c < 0 && c > -1e-100)) {
            printf("Re-normalizing\n");
            for(j=0;j<nVars;j++) {
                w[j] = c*w[j];
            }
            c = 1;
        }

    }
    return;
}
