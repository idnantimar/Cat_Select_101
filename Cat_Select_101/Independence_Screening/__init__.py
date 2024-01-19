"""
Created on Thu Jan 11 23:46:38 2024

Topic: Usually variable selection methods rely on the coefficients in some parametric model of Y|X=x.
       But for Categorical Response case, Nonparametric tests for homogeneity can be applied for variable selection.
       This methods can specially work well for misspecified models.

@author: R.Nandi
"""



#### Model-Free approach to variable selection (for Continuous Predictors) ====

import numpy as np
import pandas as pd
from .. import My_Template_FeatureImportance



class SIS_importance(My_Template_FeatureImportance):
    """
        Key Idea : In the setup of categorical response Y, a feature Xj is unimportant
        if Xj|Y=y are very similar for all categories y.

        Pros : Can be beneficial for misspecified models.

        Cons : This method only uses marginal utilities, ignoring joint covariance structure
        of the features.

        Parameters
        ----------
        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        threshold : float ; default 1e-10
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

        Attribures
        ----------

        References
        ----------
        ..[1] Cui, Hengjian, Runze Li, and Wei Zhong. "Model-free feature screening
        for ultrahigh dimensional discriminant analysis."
        Journal of the American Statistical Association 110.510 (2015): 630-641.

    """

    _coef_to_importance = None
    _permutation_importance = None
    def __init__(self,*,max_features=None,threshold=1e-10):
        super().__init__()
        self.threshold = threshold
        self.max_features = max_features


    def fit(self,X,y):
        """
        ``fit`` method for ``SIS_importance``.

        This method is extremely fast as it does not involve any cross-validation or hyperparameter tuning.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples.

            [ NOTE: only valid for numerical features ]

        y : Series of shape (n_samples,)
            The target values.

        Returns
        -------
        self
            The fitted ``SIS_importance`` instance is returned.
        """
        super().fit(X,y)
        ### empirical pmf of categorical response .....
        u = pd.get_dummies(y,dtype=float,drop_first=False).to_numpy().T
                ## each row is indicator for a response class label
        class_probs = u.mean(axis=-1,keepdims=True)
        ### computation for each feature .....
        u = (u/class_probs) - 1
                ## this will be useful for calculating [F(X|Y) - F(X)]
        def for_jColumn(Xj):
            out = 0
            for x in Xj :
                v = np.mean((Xj<=x)*u,axis=-1,keepdims=True)
                    ## this quantity is [F(X|Y) - F(X)]
                out += np.sum(class_probs*(v**2),axis=None)
            return out/self.n_samples_
        ### iterating over the columns .....
        X = X.to_numpy()
        self.feature_importances_ = np.apply_along_axis(for_jColumn,
                                                        axis=0,arr=X)
        return self


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
##
###
####
###
##
#### For Categorical Predictors ===============================================


class SIScat_importance(My_Template_FeatureImportance):
    """
        Key Idea : In the setup of categorical response Y, a feature Xj is unimportant
        if Xj|Y=y are very similar for all categories y.

        Pros : Can be beneficial for misspecified models.

        Cons : This method only uses marginal utilities, ignoring joint covariance structure
        of the features.

        Parameters
        ----------
        max_features : int ; default None
            The maximum possible number of selection. None implies no constrain,
            otherwise the `threshold` will be updated automatically if it attempts to
            select more than `max_features`.

        threshold : float ; default 1e-10
            A cut-off, any feature with importance exceeding this value will be selected,
            otherwise will be rejected.

        Attribures
        ----------

        References
        ----------
        ..[1] Cui, Hengjian, Runze Li, and Wei Zhong. "Model-free feature screening
        for ultrahigh dimensional discriminant analysis."
        Journal of the American Statistical Association 110.510 (2015): 630-641.

        [ Just like CDF for continuous features, we will use PMF for categorical features ]

    """

    _coef_to_importance = None
    _permutation_importance = None
    def __init__(self,*,max_features=None,threshold=1e-10):
        super().__init__(random_state)
        self.threshold = threshold
        self.max_features = max_features


    def fit(self,X,y):
        """
        ``fit`` method for ``SIScat_importance``.

        This method is extremely fast as it does not involve any cross-validation or hyperparameter tuning.

        Parameters
        ----------
        X : DataFrame of shape (n_samples, n_features)
            The training input samples.

            [ NOTE: only valid for categorical features ]

        y : Series of shape (n_samples,)
            The target values.

        Returns
        -------
        self
            The fitted ``SIScat_importance`` instance is returned.
        """
        super().fit(X,y)
        ### empirical pmf of categorical response .....
        u = pd.get_dummies(y,dtype=float,drop_first=False).to_numpy().T
                ## each row is indicator for a response class label
        class_probs = u.mean(axis=-1,keepdims=True)
        ### computation for each feature .....
        u = (u/class_probs) - 1
                ## this will be useful for calculating [P(X|Y) - P(X)]
        def for_jColumn(Xj):
            out = 0
            Xj = pd.get_dummies(Xj,dtype=bool,drop_first=False).to_numpy().T
                ## each column is indicator for a predictor class label
            Xj_probs = Xj.mean(axis=-1)
                ## marginal pmf for Xj
            for i in range(Xj.shape[0]) :
                v = np.mean(Xj[i]*u,axis=-1,keepdims=True)
                    ## this quantity is [P(X|Y) - P(X)]
                out += np.sum(class_probs*(v**2),axis=None)*Xj_probs[i]
            return out
        ### iterating over the columns .....
        self.feature_importances_ = X.apply(for_jColumn,axis=0).to_numpy()
        return self


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
