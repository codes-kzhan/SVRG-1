#include <math.h>
#include <stdlib.h>
#include <time.h>
//#include "mex.h"
#include "/usr/local/MATLAB/R2017a/extern/include/mex.h"
#include "mkl.h"
#define DEBUG 0

/*
SIGM_sparse(w,Xt,y,lambda,eta,d,g);
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
    long i, idx, j, nVars;

    mwIndex *jc, *ir;

    double *w, *wtilde, *G, *Xt, *y, lambda, eta, *u, *z, tau1, tau2, innerProdI, innerProdZ, tmpDelta;

    double cZ = 1, cU = 1, *cumSumZG, *cumSumZW, *cumSumZU, *cumSumUG, *cumSumUZ, *cumSumUW, tmpFactor1, tmpFactor2;
    int tmpIdx1, tmpIdx2, *lastVisited;

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

    // sparse matrix uses scaling and lazy stuff
    if (mxIsSparse(prhs[3]))
    {
        jc = mxGetJc(prhs[3]);
        ir = mxGetIr(prhs[3]);
    }
    else
    {
        mexErrMsgTxt("the dataset must be sparse");
    }

    /* Allocate memory needed for lazy updates */
    lastVisited = mxCalloc(nVars,sizeof(int));
    cumSumUG = mxCalloc(maxIter,sizeof(double));
    cumSumUW = mxCalloc(maxIter,sizeof(double));
    cumSumUZ = mxCalloc(maxIter,sizeof(double));
    cumSumZG = mxCalloc(maxIter,sizeof(double));
    cumSumZW = mxCalloc(maxIter,sizeof(double));
    cumSumZU = mxCalloc(maxIter,sizeof(double));

    // @NOTE main loop
    for (i = 0; i < maxIter; i++)
    {
        //idx = rand() % nSamples;  // sample
        idx = i % nSamples;  // sample
#if DEBUG_RANDOM
        if (i == 0)
            printf("idx: %ld\n", idx);
#endif

        /* Step 1: Compute current values of needed parameters u_{i} and z_{i} */
        if (i > 0)
        {
            for(j = jc[idx]; j < jc[idx+1]; j++)
            {
                tmpIdx2 = ir[j];
                tmpIdx1 = lastVisited[tmpIdx2]-1;
                if (tmpIdx1 == -1)
                {  // or we can let lastVisited[-1] = 0
                    z[tmpIdx2] += G[tmpIdx2] * cumSumZG[i-1] + wtilde[tmpIdx2] * cumSumZW[i-1];
                    u[tmpIdx2] += G[tmpIdx2] * cumSumUG[i-1] + wtilde[tmpIdx2] * cumSumUW[i-1];
                    z[tmpIdx2] = (z[tmpIdx2] + u[tmpIdx2] * cumSumZU[i-1] * cU) / (1 - cumSumZU[i-1] * cumSumUZ[i-1] * cU * cZ);
                    u[tmpIdx2] += z[tmpIdx2] * cumSumUZ[i-1] * cZ;
                }
                else
                { // if lastVisited[ir[j]] > 0
                    tmpFactor1 = cumSumZU[i-1] - cumSumZU[tmpIdx1];
                    tmpFactor2 = cumSumUZ[i-1] - cumSumUZ[tmpIdx1];

                    z[tmpIdx2] += G[tmpIdx2] * (cumSumZG[i-1] - cumSumZG[tmpIdx1]) + wtilde[tmpIdx2] * (cumSumZW[i-1] - cumSumZW[tmpIdx1]);
                    u[tmpIdx2] += G[tmpIdx2] * (cumSumUG[i-1] - cumSumUG[tmpIdx1]) + wtilde[tmpIdx2] * (cumSumUW[i-1] - cumSumUW[tmpIdx1]);
                    z[tmpIdx2] = (z[tmpIdx2] + u[tmpIdx2] * tmpFactor1 * cU) / (1 - tmpFactor1 * tmpFactor2 * cU * cZ);
                    u[tmpIdx2] += z[tmpIdx2] * tmpFactor2 * cZ;
                }
                lastVisited[tmpIdx2] = i;
            }
        }

        /* Step 2: lazily update w_{i+1} */
        for(j = jc[idx]; j < jc[idx+1]; j++)
        {
            tmpIdx2 = ir[j];
            w[tmpIdx2] = (1 - tau1 - tau2) * u[tmpIdx2] * cU + tau2 * wtilde[tmpIdx2] + tau1 * z[tmpIdx2] * cZ;
        }

        /* Step 3: Compute derivative of loss \nabla f(w_{i}) */
        innerProdI = 0;
        innerProdZ = 0;
        for(j = jc[idx]; j < jc[idx+1]; j++)
        {
            innerProdI += w[ir[j]] * Xt[j];
            innerProdZ += wtilde[ir[j]] * Xt[j];
        }
        tmpDelta = -y[idx] / (1 + exp(y[idx] * innerProdI)) + y[idx] / (1 + exp(y[idx] * innerProdZ));

        // update cumSum
        cZ *= 1 - eta * tau1 * lambda;
        cU *= (1 - tau1 - tau2) * (1 - eta * tau1 * lambda);

        if (i == 0)
        {
            cumSumUG[0] = - eta * tau1 / cU;
            cumSumUZ[0] = tau1 * (1 - eta * tau1 * lambda) / cU;
            cumSumUW[0] = tau2 * (1 - eta * tau1 * lambda) / cU;
            cumSumZG[0] = - eta / cZ;
            cumSumZU[0] = - eta * lambda * (1 - tau1 - tau2) / cZ;
            cumSumZW[0] = - eta * lambda * tau2 / cZ;
        }
        else
        {
            cumSumUG[i] = cumSumUG[i-1] - eta * tau1 / cU;
            cumSumUZ[i] = cumSumUZ[i-1] + tau1 * (1 - eta * tau1 * lambda) / cU;
            cumSumUW[i] = cumSumUW[i-1] + tau2 * (1 - eta * tau1 * lambda) / cU;
            cumSumZG[i] = cumSumZG[i-1] - eta / cZ;
            cumSumZU[i] = cumSumZU[i-1] - eta * lambda * (1 - tau1 - tau2) / cZ;
            cumSumZW[i] = cumSumZW[i-1] - eta * lambda * tau2 / cZ;
        }

        /* Step 4: approximate z_{i+1} and z_{i+1} */
        for(j = jc[idx]; j < jc[idx+1]; j++)
        {
            z[ir[j]] -= eta / cZ * tmpDelta * Xt[j];
            u[ir[j]] -= eta * tau1 / cU * tmpDelta * Xt[j];
        }

        // Re-normalize the parameter vector if it has gone numerically crazy
        if((cZ > 1e100 || cZ < -1e100 || (cZ > 0 && cZ < 1e-100) || (cZ < 0 && cZ > -1e-100)) || (cU > 1e100 || cU < -1e100 || (cU > 0 && cU < 1e-100) || (cU < 0 && cU > -1e-100)))
        {
#if DEBUG
            mexPrintf("Oop, we have to re-normalize, i: %d\n", i);
#endif
            for(j = 0; j < nVars; j++)
            {
                tmpIdx2 = j;
                tmpIdx1 = lastVisited[j]-1;
                if (lastVisited[j] == 0)
                {
                    z[tmpIdx2] += G[tmpIdx2] * cumSumZG[i] + wtilde[tmpIdx2] * cumSumZW[i];
                    u[tmpIdx2] += G[tmpIdx2] * cumSumUG[i] + wtilde[tmpIdx2] * cumSumUW[i];
                    z[tmpIdx2] = (z[tmpIdx2] + u[tmpIdx2] * cumSumZU[i] * cU) / (1 - cumSumZU[i] * cumSumUZ[i] * cZ * cU);
                    u[tmpIdx2] += z[tmpIdx2] * cumSumUZ[i] * cZ;
                }
                else
                {
                    tmpFactor1 = cumSumZU[i] - cumSumZU[tmpIdx1];
                    tmpFactor2 = cumSumUZ[i] - cumSumUZ[tmpIdx1];

                    z[tmpIdx2] += G[tmpIdx2] * (cumSumZG[i] - cumSumZG[tmpIdx1]) + wtilde[tmpIdx2] * (cumSumZW[i] - cumSumZW[tmpIdx1]);
                    u[tmpIdx2] += G[tmpIdx2] * (cumSumUG[i] - cumSumUG[tmpIdx1]) + wtilde[tmpIdx2] * (cumSumUW[i] - cumSumUW[tmpIdx1]);
                    z[tmpIdx2] = (z[tmpIdx2] + u[tmpIdx2] * tmpFactor1 * cU) / (1 - tmpFactor1 * tmpFactor2 * cZ * cU);
                    u[tmpIdx2] += z[tmpIdx2] * tmpFactor2 * cZ;
                }
                lastVisited[j] = i+1;
            }
            cumSumZG[i] = 0;
            cumSumZW[i] = 0;
            cumSumZU[i] = 0;
            cumSumUG[i] = 0;
            cumSumUW[i] = 0;
            cumSumUZ[i] = 0;
            cblas_dscal(nVars, cZ, z, 1);
            cblas_dscal(nVars, cU, u, 1);
            cZ = 1;
            cU = 1;
        }

    }

    i = maxIter - 1;
    for(j = 0; j < nVars; j++)
    {
        tmpIdx2 = j;
        tmpIdx1 = lastVisited[j]-1;
        if (lastVisited[j] == 0)
        {
            z[tmpIdx2] += G[tmpIdx2] * cumSumZG[i] + wtilde[tmpIdx2] * cumSumZW[i];
            u[tmpIdx2] += G[tmpIdx2] * cumSumUG[i] + wtilde[tmpIdx2] * cumSumUW[i];
            z[tmpIdx2] = (z[tmpIdx2] + u[tmpIdx2] * cumSumZU[i] * cU) / (1 - cumSumZU[i] * cumSumUZ[i] * cZ * cU);
            u[tmpIdx2] += z[tmpIdx2] * cumSumUZ[i] * cZ;
        }
        else
        {
            tmpFactor1 = cumSumZU[i] - cumSumZU[tmpIdx1];
            tmpFactor2 = cumSumUZ[i] - cumSumUZ[tmpIdx1];

            z[tmpIdx2] += G[tmpIdx2] * (cumSumZG[i] - cumSumZG[tmpIdx1]) + wtilde[tmpIdx2] * (cumSumZW[i] - cumSumZW[tmpIdx1]);
            u[tmpIdx2] += G[tmpIdx2] * (cumSumUG[i] - cumSumUG[tmpIdx1]) + wtilde[tmpIdx2] * (cumSumUW[i] - cumSumUW[tmpIdx1]);
            z[tmpIdx2] = (z[tmpIdx2] + u[tmpIdx2] * tmpFactor1 * cU) / (1 - tmpFactor1 * tmpFactor2 * cZ * cU);
            u[tmpIdx2] += z[tmpIdx2] * tmpFactor2 * cZ;
        }
        lastVisited[j] = i+1;
    }

    cblas_dscal(nVars, cZ, z, 1);
    cblas_dscal(nVars, cU, u, 1);

    mxFree(lastVisited);
    mxFree(cumSumZG);
    mxFree(cumSumZU);
    mxFree(cumSumZW);
    mxFree(cumSumUG);
    mxFree(cumSumUZ);
    mxFree(cumSumUW);
    return;
}
