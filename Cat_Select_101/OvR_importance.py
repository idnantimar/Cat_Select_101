"""
Created on Tue Jan 16 19:39:48 2024

Topic: One-vs-Rest classification based feature selection.

@author: R.Nandi
"""



#### OvR importance ===========================================================

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from . import My_Template_FeatureImportance




class OvR_importance(My_Template_FeatureImportance):
    """
        Feature selection using the One-vs-Rest classification scheme.

        Key Idea : Each OvR classifier is a binary Logistic Regression. So absolute
        values of `coef_` can be considered as feature-importances corresponding to
        that specific target class. Importances from all OvR classifiers will be aggregated
        for combined `feature_importances_`


        Class Variables
        ---------------
        Estimator_Type : ``{'penalty':'l1','multi_class':'ovr','solver':'saga'}``


        Parameters
        ----------
        random_state : int ; default None
            Seed for reproducible results across multiple function calls.

        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        threshold : float ; default 1e-10
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

        Attribures
        ----------

    """
    Estimator_Type = {'penalty':'l1','multi_class':'ovr','solver':'saga'}
    _coef_to_importance = None
    _permutation_importance = None
    def __init__(self,random_state=None,*,
                 max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.threshold = threshold
        self.max_features = max_features


    def fit(self,X,y,Cs=[1.0],*,max_iter=1000,**kwargs):
        """
        ``fit`` method for ``OvR_importance``.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples. If a DataFrame with categorical column is passed as input,
            `n_features_in_` will be number of columns after ``pd.get_dummies(X,drop_first=True)`` is applied.

        y : Series of shape (n_samples,)
            The target values.

        Cs : list of floats ; default [1.0]
            The inverse of regularization strength.

        max_iter : int ; default 100
            Maximum number of iterations of the optimization algorithm.

        **kwargs : other keyword arguments to ``LogistiRegression()`` or ``LogistiRegressionCV()``

        Returns
        -------
        self
            The fitted ``OvR_importance`` instance is returned.

        """
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        super().fit(X,y)
        ## fitting the model .....
        if len(Cs)>1 :
            estimator = LogisticRegressionCV(**self.Estimator_Type,
                                             Cs=Cs,
                                             random_state=self.random_state,max_iter=max_iter,
                                             **kwargs)
        else :
            estimator = LogisticRegression(**self.Estimator_Type,
                                           C=Cs[0],
                                           random_state=self.random_state,max_iter=max_iter,
                                           **kwargs)
        estimator.fit(X,y)
        ## feature_importances .....
        self.estimator = estimator
        self.coef_ = estimator.coef_
        self.intercept_ = estimator.intercept_
        self.class_specific_importances_ = np.abs(self.coef_)
        self.feature_importances_ = self.class_specific_importances_.mean(axis=0)
        return self


    def transform(self,X):
        X = pd.get_dummies(X,drop_first=True,dtype=int)
        return super().transform(X)


    def get_error_rates(self,true_coef,*,plot=False):
        """
        Computes various error-rates when true importance of the features are known.

        Parameters
        ----------
        true_coef : array of shape (`n_features_in_`,)
            If a boolean array , True implies the feature is important in true model, null feature otherwise.
            If a array of floats , it represent the `feature_importances_` of the true model.

        plot : bool ; default False
            Whether to plot the `confusion_matrix_for_features_`.

        Returns
        -------
        dict
            Returns the empirical estimate of various error-rates
           {'PCER': per-comparison error rate,
            'FDR': false discovery rate,
            'PFER': per-family error rate,
            'TPR': true positive rate
            }

        """
        self.get_support()
        self.true_coef = np.array(true_coef)
        if (self.true_coef.dtype==bool) :
            self.true_support = self.true_coef
        else :
            self.true_support = (self.true_coef >= self.threshold_)
        return super().get_error_rates(plot=plot)



#### ==========================================================================
