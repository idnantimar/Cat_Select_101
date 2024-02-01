"""
Created on Thu Feb  1 13:32:51 2024

Topic: Boxplot Cutoff Thresholding.

@author: R.Nandi
"""



#### BCT ======================================================================

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator,TransformerMixin




class BCT_selection(TransformerMixin,BaseEstimator):
    """


        References
        ----------
        ..[1] Akarachantachote, Noppamas, Seree Chadcham, and Kidakan Saithanu.
        "Cutoff threshold of variable importance in projection for variable selection."
        Int J Pure Appl Math 94.3 (2014): 307-322.

    """

    def __init__(self,base_estimator,*,random_state=None):
        self.base_estimator = base_estimator
        self.random_state = random_state


    def _augment_noise(self,X):
        rng = np.random.default_rng(self.random_state)
        shuffle = lambda x : rng.permutation(x)
        X_noise = X.apply(shuffle,axis=0)
        X.columns = X.columns.astype(str)
        X_noise.columns = [name+'_noise' for name in X.columns]
        return pd.concat([X,X_noise],axis=1)


    def fit(self,X,y,**fit_params):
        ## adding noise matrix ....
        Z,y = self._augment_noise(pd.DataFrame(X)),pd.Series(y)
        ## fitting base estimator ....
        self.base_estimator.fit(Z,y,**fit_params)
        self.n_samples_,self.n_features_in_ = (self.base_estimator.n_samples_,
                                               self.base_estimator.n_features_in_//2)
        importances_ = self.base_estimator.feature_importances_.reshape((2,-1))
                        # 1st row is important, 2nd row is noise in `importances_`
        self.feature_importances_ = importances_[0]
        self.ranking_ = rankdata(-self.feature_importances_,method='ordinal')
        self.feature_names_in_ = self.base_estimator.feature_names_in_[:self.n_features_in_]
        ## BCT = _Q3 + 1.5(_Q3-_Q1) ....
        _Q1,_Q3 = np.quantile(importances_[1],0.25),np.quantile(importances_[1],0.75)
        self.BCT = 2.5*_Q3 - 1.5*_Q1
        ## remove noise from base estimator ....
        reset = {'n_features_in_':self.n_features_in_,
                 'feature_importances_':self.feature_importances_,
                 'threshold_':self.BCT,
                 'ranking_':self.ranking_,
                 'feature_names_in_':self.feature_names_in_}
        for key,value in reset.items() :
            setattr(self.base_estimator,key,value)
        return self


    def transform(self,X):
        return self.base_estimator.transform(X)




#### ==========================================================================
