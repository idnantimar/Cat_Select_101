"""
Created on Wed Jan 17 19:03:38 2024

Topic: Least Squares with slack variables for classification task.

@author: R.Nandi
"""



#### ==========================================================================

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted




class d_LSR(BaseEstimator):
    """
    Discriminative Least Squares Regression for multiclass classification.

    Parameters
    ----------
    regularization : float ; default 0.1
        The strength of regularization.

    u : float ; default 1e+4
        A large positive number (virtually +ve infinity).

    References
    ----------
    ..[1] Xiang, Shiming, et al. "Discriminative least squares regression for multiclass
    classification and feature selection." IEEE transactions on neural networks and learning
    systems 23.11 (2012): 1738-1754.

    """
    def __init__(self,regularization=0.1,u=1e+4):
        self.regularization = regularization
        self.u = u


    def fit(self,X,y,*,
            max_iter=30,tol=1e-4):
        """
        ``fit`` method for ``d_LSR``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,n_classes)
            The one-hot encoded target values.

        max_iter : int ; default 30
            The maximum number of iterations.

        tol : float ; default 1e-4
            The tolerence for convergence criterion.

        Returns
        -------
        self

        Attributes
        ----------
        coef_ : array of shape (n_features,n_classes)
            The estimated coefficients.

        intercept_ : array of shape (1,n_classes)
            The estimated intercepts.
        """
        ### initialization ... ...
        n,m = X.shape
        c = y.shape[1]
        M = np.zeros_like(y,dtype=float)
        W0 = np.zeros((m,c),dtype=float)
        t0 = np.zeros((1,c),dtype=float)
        _W0 = np.concatenate([W0,t0],axis=0)
        B = np.where(y,1,-1)
        _X = np.concatenate([X,np.full((n,1),self.u)],axis=1)
        ### Iterative Updates ... ...
        ## Outer loop ....
        for k in range(max_iter):
            T = y + B*M
            SIGMA,D = np.eye(m+1,dtype=float),np.eye(n,dtype=float)
            ## Inner loop start ...
            for _t in range(max_iter):
                _XtD = np.matmul(_X.T,D)
                _W = np.linalg.solve(np.matmul(_XtD,_X) + self.regularization*SIGMA,
                                    np.matmul(_XtD,T))
                if np.linalg.norm(_W-_W0,ord='fro')**2<tol:
                    break
                else:
                    SIG_ii = 1/np.linalg.norm(_W,ord=2,axis=1)
                    SIGMA = np.diag(SIG_ii)
                    D_ii = 1/np.linalg.norm(np.matmul(_X,_W)-y,ord=2,axis=1)
                    D = np.diag(D_ii)
                    _W0 = _W
            ## Inner loop end |
            W = _W[:-1]
            t = self.u*_W[-1:]
            if (np.linalg.norm(W-W0,ord='fro')**2 + np.linalg.norm(t-t0,ord=2)**2) < tol:
                break
            else:
                BP = B*((np.matmul(X,W) + t) - y)
                M = np.where(BP>0,BP,0)
                W0,t0 = W,t
        ## Outer loop end ||
        ### return values ... ...
        self.coef_ = W
        self.intercept_ = t
        return self


    def predict(self,X):
        """
        ``predict`` method for ``d_LSR``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        array of shape (n_samples,)
            The predicted class labels are returned.

        """
        check_is_fitted(self)
        predicted = np.matmul(X,self.coef_) + self.intercept_
        return predicted.argmax(axis=1)


    def score(self,X,y):
        """
        ``score`` method for ``d_LSR``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,n_classes)
            The one-hot encoded target values.

        Returns
        -------
        float
            The ``accuracy`` metric.

        """
        compare = (self.predict(X)==y.argmax(axis=1))
        return np.mean(compare,axis=None)



#### ==========================================================================
