#include <math.h>
#include "mex.h"

/*
IAG_logistic(w,Xt,y,lambda,alpha,d,g);
% w(p,1) - updated in place
% Xt(p,n) - real, can be sparse
% y(n,1) - {-1,1}
% lambda - scalar regularization param
% stepSize - scalar constant step size
% who cares % iVals(maxIter,1) - sequence of examples to choose
% The below are updated in place and are needed for restarting the algorithm
% d(p,1) - initial approximation of average gradient (should be sum of previous gradients)
% g(n,1) - previous derivatives of loss
% maxIter - scalar maximal iterations
% who cares % covered(n,1) - whether the example has been visited
*/

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    /* Variables */
    int k,nSamples,maxIter,sparse=0,useScaling=1,useLazy=1,*lastVisited;
    long i,j,nVars;

    mwIndex *jc,*ir;

    double *w, *Xt, *y, lambda, alpha, innerProd, sig,c=1,*g,*d,*cumSum;

    if (nrhs != 8)
        mexErrMsgTxt("Function needs 8 arguments: {w,Xt,y,lambda,alpha,d,g}");

    /* Input */

    w = mxGetPr(prhs[0]);
    Xt = mxGetPr(prhs[1]);
    y = mxGetPr(prhs[2]);
    lambda = mxGetScalar(prhs[3]);
    alpha = mxGetScalar(prhs[4]);
    d = mxGetPr(prhs[5]);
    g = mxGetPr(prhs[6]);

    /* Compute Sizes */
    nVars = mxGetM(prhs[1]);
    nSamples = mxGetN(prhs[1]);
    maxIter = (int)mxGetScalar(prhs[7]);
    //printf("maxIter: %d\n", maxIter);

    if (nVars != mxGetM(prhs[0]))
        mexErrMsgTxt("w and Xt must have the same number of rows");
    if (nSamples != mxGetM(prhs[2]))
        mexErrMsgTxt("number of columns of Xt must be the same as the number of rows in y");
    if (nVars != mxGetM(prhs[5]))
        mexErrMsgTxt("Xt and d must have the same number of rows");
    if (nSamples != mxGetM(prhs[6]))
        mexErrMsgTxt("Xt and g must have the same number of columns");

    // sparse matrix uses scaling and lazy stuff
    if (mxIsSparse(prhs[1])) {
        sparse = 1;
        jc = mxGetJc(prhs[1]);
        ir = mxGetIr(prhs[1]);
    }
    else {
        useScaling = 0;
        useLazy = 0;
    }

    if (sparse && alpha*lambda==1)
        mexErrMsgTxt("Sorry, I don't like it when Xt is sparse and alpha*lambda=1\n");
        // why not?

    /* Allocate memory needed for lazy updates */
    if (useLazy) {
        lastVisited = mxCalloc(nVars,sizeof(int));
        cumSum = mxCalloc(maxIter,sizeof(double));

        /*for(j=0;j<nVars;j++)
            lastVisited[j] = -1;*/
    }

    for(k = 0; k < maxIter; k++)
    {
        /* Select next training example */
        i = k % nSamples;

        /* Compute current values of needed parameters */
        if (useLazy && k > 0) {
            for(j=jc[i];j<jc[i+1];j++) {
                if (lastVisited[ir[j]]==0) {  // or we can let lastVisited[-1] = 0
                    w[ir[j]] -= d[ir[j]]*cumSum[k-1];
                }
                else { // if lastVisited[ir[j]] > 0
                    w[ir[j]] -= d[ir[j]]*(cumSum[k-1]-cumSum[lastVisited[ir[j]]-1]);
                }
                lastVisited[ir[j]] = k;
            }
        }

        /* Compute derivative of loss */
        innerProd = 0;
        if (sparse) {
            for(j=jc[i];j<jc[i+1];j++)
                innerProd += w[ir[j]]*Xt[j];
            if (useScaling)
                innerProd *= c;
        }
        else {
            for(j=0;j<nVars;j++)
                innerProd += w[j]*Xt[j + nVars*i];
        }
        sig = -y[i]/(1+exp(y[i]*innerProd));

        /* Update direction */
        if (sparse) {
            for(j=jc[i];j<jc[i+1];j++)
                d[ir[j]] += Xt[j]*(sig - g[i]);
        }
        else {
            for(j=0;j<nVars;j++)
                d[j] += Xt[j + nVars*i]*(sig - g[i]);
        }

        /* Store derivative of loss */
        g[i] = sig;

        /* Update parameters */
        if (useScaling)
        {
            c *= 1-alpha*lambda;

            if (useLazy) {
                if (k==0)
                    cumSum[0] = alpha/(c*nSamples);
                else
                    cumSum[k] = cumSum[k-1] + alpha/(c*nSamples);
            }
            else {
                for(j=0;j<nVars;j++)
                    w[j] -= alpha*d[j]/(c*nSamples);
            }
        }
        else {
            for(j=0;j<nVars;j++) {
                w[j] *= 1-alpha*lambda;
            }

            for(j=0;j<nVars;j++)
                w[j] -= alpha*d[j]/nSamples;
        }

         /* Re-normalize the parameter vector if it has gone numerically crazy */
        if(c > 1e100 || c < -1e100 || (c > 0 && c < 1e-100) || (c < 0 && c > -1e-100)) {
            printf("Re-normalizing\n");

            if (useLazy) {
                for(j=0;j<nVars;j++) {
                    if (lastVisited[j]==0)
                        w[j] -= d[j]*cumSum[k];
                    else
                        w[j] -= d[j]*(cumSum[k]-cumSum[lastVisited[j]-1]);
                    lastVisited[j] = k+1;
                }
                cumSum[k] = 0;
            }
            for(j=0;j<nVars;j++) {
                w[j] = c*w[j];
            }
            c = 1;
        }

    }

    if (useLazy) {
        for(j=0;j<nVars;j++) {
            if (lastVisited[j]==0) {
                w[j] -= d[j]*cumSum[maxIter-1];
            }
            else
            {
                w[j] -= d[j]*(cumSum[maxIter-1]-cumSum[lastVisited[j]-1]);
            }
        }
    }

    if(useScaling) {
        for(j=0;j<nVars;j++)
            w[j] *= c;
    }

    if(useLazy) {
        mxFree(lastVisited);
        mxFree(cumSum);
    }
}
