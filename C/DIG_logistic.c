#include <math.h>
#include "mex.h"
#define DEBUG 0

/*
 * SGD_logistic(w,Xt,y,lambda,stepSizes,iVals,maxNorm,average);
 w(p,1) - updated in place
 Xt(p,n) - real, can be sparse
 y(n,1) - {-1,1}
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

    double *w, *Xt, *y, lambda, innerProd, alpha, factor, sig, c=1;

    /* Input */

    if (nrhs < 7)
        mexErrMsgTxt("At least 6 arguments are needed: {w, Xt, y, lambda, passes, maxIter, factor}");

    w = mxGetPr(prhs[0]);
    Xt = mxGetPr(prhs[1]);
    y = mxGetPr(prhs[2]);
    lambda = mxGetScalar(prhs[3]);
    passes = (int)mxGetScalar(prhs[4]);
    maxIter = (int)mxGetScalar(prhs[5]);
    factor = mxGetScalar(prhs[6]);

    /* Compute Sizes */
    nVars = mxGetM(prhs[1]);
    nSamples = mxGetN(prhs[1]);

    if (nVars != mxGetM(prhs[0]))
        mexErrMsgTxt("w and Xt must have the same number of rows");
    if (nSamples != mxGetM(prhs[2]))
        mexErrMsgTxt("number of columns of Xt must be the same as the number of rows in y");

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
                innerProd += w[ir[j]]*Xt[j];
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
                innerProd += w[j] * Xt[j + nVars*i];
            }
        }
        sig = -y[i]/(1+exp(y[i]*innerProd));

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
                    w[ir[j]] -= alpha*Xt[j]*sig/c;
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
                    w[ir[j]] -= alpha*Xt[j]*sig;
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
                w[j] -= alpha*Xt[j + i*nVars]*sig;
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
