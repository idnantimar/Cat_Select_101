"""
Created on Sun Dec 31 23:10:21 2023

Topic: Stabilty Selection is not any particular method,
        but a framework that can be applied on any base method for reduced variability on selected support.

Note : An implementation of Stability selection already available at https://github.com/scikit-learn-contrib/stability-selection .

(i) This one is just a bit customized as per our rest of the codes,
since we have already implemented ``get_support()`` in our `base_estimator`.

(ii) Instead of entirely dropping informations about individual iterations,
we stored `selected_over_subsamples_` as csr matrix for any further use.

(iii) Instead of computing `support_` for a scalar tau, option for computing
support for multiple tau values at a time, to avoid redundant calculations.


@author: R.Nandi
"""




#### Complementary Pairs Stability Selection ===================================

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin,clone
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from ..vanilla_LASSO import vanillaLASSO_importance




class CPSS(TransformerMixin,BaseEstimator):
    """
        'Complementary Pairs Stability Selection' fits the `base_estimator` several times on
        two randomly chosen pairs of subsets of size [n_samples/2] each,
        for possibly different values of the regularization parameter. Variables that
        reliably get selected by the model in these iterations are
        considered to be stable variables.


    References
    ----------
    .. [1] Shah, R.D. and Samworth, R.J., 2013. Variable selection with
           error control: another look at stability selection. Journal
           of the Royal Statistical Society: Series B (Statistical Methodology),
            75(1), pp.55-80.

    .. [2] Meinshausen, N. and Buhlmann, P., 2010. Stability selection.
           Journal of the Royal Statistical Society: Series B
           (Statistical Methodology), 72(4), pp.417-473.

    """
    def __init__(self,base_estimator=vanillaLASSO_importance(Cs=[1]),n_resamples=20,*,
                 kwargs_Parallel={'n_jobs':None},random_state=None):
        self.base_estimator = base_estimator
        self.n_resamples = n_resamples
        self.kwargs_Parallel = kwargs_Parallel
        self.random_state = random_state


    def _subsamples(self,n,rng):
        all_samples = range(n)
        n_ = (n//2)*2
        subsamples = np.empty((2*self.n_resamples,n//2),dtype=int)
        for i in range(self.n_resamples):
            current = rng.permutation(all_samples)[:n_]
            subsamples[2*i:2*(i+1)] = current.reshape((2,-1))
        return subsamples


    def fit(self,X,y,**fit_params):
        X,y = pd.DataFrame(X),pd.Series(y)
        self.n_samples_ = X.shape[0]
        ### Generating complementary pairs .....
        subsamples = self._subsamples(n=self.n_samples_,
                                      rng=np.random.default_rng(self.random_state))
        ### Feature Selection/Rejection for each subsample of size [n/2] .....
        def for_a_subsample(index):
            base_estimator = clone(self.base_estimator)
            base_estimator.fit(X.iloc[index],y.iloc[index],**fit_params)
            return base_estimator.get_support()
        ### The main computation chunk .....
        self.selected_over_subsamples_ = np.array(Parallel(**self.kwargs_Parallel)(
            delayed(for_a_subsample)(row) for row in subsamples))
        ### Stability Scores .....
        self.n_features_in_ = self.selected_over_subsamples_.shape[1]
        self.stability_scores_ = self.selected_over_subsamples_.mean(axis=0)
        self.selected_over_subsamples_ = csr_matrix(self.selected_over_subsamples_)
        return self


    def get_support(self,tau=0.6):
        self.tau = np.array(tau).T
        self.support_ = (self.stability_scores_ >= self.tau)
        return self.support_

